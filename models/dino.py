from collections import defaultdict
from functools import partial
from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from torch import nn

from models.utils import sinkhorn_knopp, momentum_update, dist_to_similarity

class ProtoDINO(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, backbone: nn.Module, adapter: nn.Module | None, pooling_method: str, cls_head: str,
                 *, metric: str = "cos", gamma: float = 0.99, n_prototypes: int = 5, n_classes: int = 200,
                 pca_fg_cmp: str = "le", pca_fg_threshold: float = 0.5, dim: int = 768):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        self.pca_fg_threshold = pca_fg_threshold
        self.backbone = backbone
        self.adapter = adapter
        
        assert pca_fg_cmp in ["ge", "le"]
        self.cmp_fg = partial(torch.ge if pca_fg_cmp =="ge" else torch.le, other=pca_fg_threshold)
        
        assert metric in ["l2", "cos"]
        self.metric = metric
        self.dim = dim
        self.register_buffer("prototypes", torch.empty(self.C, self.n_prototypes, self.dim))

        nn.init.trunc_normal_(self.prototypes, std=0.02)
        for param in self.adapter.parameters():
            nn.init.zeros_(param)
        
        assert pooling_method in ["sum", "avg", "max"]
        assert cls_head in ["fc", "sum", "avg", 'sa']
        self.pooling_method = pooling_method
        self.cls_head = cls_head

        if cls_head == "fc":
            self.fc = nn.Linear(self.n_prototypes * self.n_classes, self.n_classes, bias=False)
            prototype_class_assiciation = torch.eye(self.n_classes).repeat_interleave(self.n_prototypes, dim=0)
            self.fc.weight = nn.Parameter((prototype_class_assiciation - 0.5 * (1 - prototype_class_assiciation)).t())
            self.sa = None
        elif cls_head == "sa":
            self.fc = None
            self.sa = nn.Parameter(torch.full((self.n_classes, self.n_prototypes,), 0.5, dtype=torch.float32))
        else:
            self.fc = None
            self.sa = None
        
        self.freeze_prototypes = False

    def update_prototypes(self,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          labels: torch.Tensor,
                          patch_labels: torch.Tensor,
                          debug: bool = False,
                          use_gumbel: bool = False):
        B, H, W = patch_labels.shape
        patch_labels_flat = patch_labels.flatten()  # shape: [B*H*W,]
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")
        
        P_old = self.prototypes.clone()
        
        debug_dict = defaultdict(dict) if debug else None
        assignment_masks = torch.empty_like(patch_labels_flat)

        for c_i, c in enumerate(labels.unique().tolist()):
            class_fg_mask = patch_labels_flat == c  # shape: [B*H*W,]
            I_c = patches_flat[class_fg_mask]  # shape: [N, dim]
            L_c = L[class_fg_mask, c, :]  # shape: [N, K,]
            
            L_c_assignment, L_c_assignment_indices = sinkhorn_knopp(L_c, use_gumbel=use_gumbel)  # shape: [N, K,], [N,]
            
            P_c_new = torch.mm(L_c_assignment.t(), I_c)  # shape: [K, dim]
            
            P_c_old = P_old[c, :, :]
            
            if self.training and (not self.freeze_prototypes):
                self.prototypes[c, ...] = momentum_update(P_c_old, P_c_new, momentum=self.gamma)
            
            if debug:
                debug_dict["L_c"][c] = L_c
                debug_dict["L_c_assignment"][c] = L_c_assignment
            
            assignment_masks[class_fg_mask] = L_c_assignment_indices + (self.n_prototypes * c_i)
        
        assignment_masks = rearrange(assignment_masks, "(B H W) -> B H W", B=B, H=H, W=W)
        
        if debug:
            return assignment_masks, dict(debug_dict)

        return assignment_masks, None
                

    def get_pseudo_patch_labels(self, patch_tokens: torch.Tensor, labels: torch.Tensor):
        B, n_patches, dim = patch_tokens.shape
        H = W = int(sqrt(n_patches))
        U, _, _ = torch.pca_lowrank(patch_tokens.reshape(-1, self.dim),
                                    q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_patch_labels = torch.where(self.cmp_fg(U_scaled),
                                          repeat(labels, "B -> B H W", H=H, W=W),
                                          self.C - 1)
        
        return pseudo_patch_labels.to(dtype=torch.long)  # shape: [B, H, W,]

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *, debug: bool = False, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)
        
        patch_tokens = self.backbone(x)  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)

        patch_tokens_updated = patch_tokens_norm + self.adapter(patch_tokens_norm)

        if self.metric == "l2":
            patch_prototype_dists = torch.cdist(patch_tokens_updated, rearrange(self.prototypes, "C K dim -> (C K) dim"), p=2)
            patch_prototype_logits = dist_to_similarity(patch_prototype_dists)
            patch_prototype_logits = rearrange(patch_prototype_logits, "B n_patches (C K) -> B n_patches C K", C=self.C, K=self.n_prototypes)
        elif self.metric == "cos":
            patch_prototype_logits = einsum(patch_tokens_updated, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
        
        if labels is not None:
            pseudo_patch_labels = self.get_pseudo_patch_labels(patch_tokens.detach(), labels=labels)
            assignment_masks, debug_dict = self.update_prototypes(
                patch_tokens=patch_tokens_norm.detach(),
                patch_prototype_logits=patch_prototype_logits.detach(),
                labels=labels,
                patch_labels=pseudo_patch_labels,
                debug=debug,
                use_gumbel=use_gumbel)
        
        if self.pooling_method == "max":
            image_prototype_logits, _ = patch_prototype_logits.max(1)  # shape: [B, C, K,], C=n_classes+1
        elif self.pooling_method == "sum":
            image_prototype_logits = patch_prototype_logits.sum(1)  # shape: [B, C, K,], C=n_classes+1
        else:
            image_prototype_logits = patch_prototype_logits.mean(1)  # shape: [B, C, K,], C=n_classes+1
        
        if self.cls_head == "sa":
            sa_weights = F.softmax(self.sa, dim=-1) * self.n_prototypes
            image_prototype_logits_weighted = image_prototype_logits[:, :-1, :] * sa_weights
            class_logits = image_prototype_logits_weighted.sum(-1)
        elif self.cls_head == "fc":
            image_prototype_logits_flat = rearrange(image_prototype_logits[:, :-1, :],
                                                    "B n_classes K -> B (n_classes K)")
            class_logits = self.fc(image_prototype_logits_flat.detach())  # shape: [B, n_classes,]
        elif self.cls_head == "mean":
            class_logits = image_prototype_logits.mean(-1)  # shape: [B, C,]
            class_logits = class_logits[:, :-1]
        else:
            class_logits = image_prototype_logits.sum(-1)  # shape: [B, C,]
            class_logits = class_logits[:, :-1]

        outputs =  dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits,  # shape: [B, n_classes,]
            assignment_masks=assignment_masks  # shape: [B, H, W,]
        )

        if labels is not None:
            outputs["pseudo_patch_labels"] = pseudo_patch_labels
            if debug:
                outputs.update(debug_dict)

        return outputs


class ProtoPNetLoss(nn.Module):
    def __init__(self, l_clst_coef: float, l_sep_coef: float, l_l1_coef: float) -> None:
        super().__init__()
        assert l_clst_coef <= 0. and l_sep_coef>= 0.
        self.l_clst_coef = l_clst_coef
        self.l_sep_coef = l_sep_coef
        self.l_l1_coef = l_l1_coef
        self.xe = nn.CrossEntropyLoss()

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor], batch: dict[str, torch.Tensor]):
        logits, dists = outputs["class_logits"], outputs["image_prototype_logits"]
        _, labels, _, _ = batch

        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        if self.l_clst_coef != 0 and self.l_sep_coef != 0:
            l_clst, l_sep = self.compute_costs(dists,labels)
            loss_dict["l_clst"] = self.l_clst_coef * l_clst
            loss_dict["l_sep"] = self.l_sep_coef * l_sep
            loss_dict["_l_clst_raw"] = l_clst
            loss_dict["_l_sep_raw"] = l_sep

        return loss_dict

    @staticmethod
    def compute_costs(l2_dists: torch.Tensor, labels: torch.Tensor):
        positives = F.one_hot(labels, num_classes=201).to(dtype=torch.float32)
        negatives = 1 - positives
        cluster_cost = torch.mean(l2_dists.max(-1).values * positives)
        separation_cost = torch.mean(l2_dists.max(-1).values * negatives)

        return cluster_cost, separation_cost
