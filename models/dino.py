from collections import defaultdict
from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from torch import nn

from models.utils import (
    sinkhorn_knopp,
    momentum_update,
)

class ProtoDINO(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, pooling_method: str, cls_head: str, *, gamma: float = 0.99, n_prototypes: int = 5, n_classes: int = 200,
                 pca_fg_threshold: float = 0.5, dim: int = 768):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        self.pca_fg_threshold = pca_fg_threshold
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        self.dim = dim
        self.register_buffer("prototypes", torch.zeros(self.C, self.n_prototypes, self.dim))

        nn.init.trunc_normal_(self.prototypes, std=0.02)
        
        self.pretrain_prototypes = True
        
        assert pooling_method in ["sum", "avg", "max"]
        self.pooling_method = pooling_method
        assert cls_head in ["fc", "sum", "avg"]
        self.cls_head = cls_head
        if cls_head == "fc":
            self.fc = nn.Linear(self.n_prototypes * self.n_classes, self.n_classes, bias=False)
            prototype_class_assiciation = torch.eye(self.n_classes).repeat_interleave(self.n_prototypes, dim=0)
            self.fc.weight = nn.Parameter((prototype_class_assiciation - 0.5 * (1 - prototype_class_assiciation)).t())
        else:
            self.fc = None
        
        self.freeze_prototypes = False

    def update_prototypes(self,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          labels: torch.Tensor,
                          patch_labels: torch.Tensor,
                          debug: bool = False,
                          use_gumbel: bool = False):
        patch_labels_flat = patch_labels.flatten()  # shape: [B*H*W,]
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")
        
        P_old = self.prototypes.clone()
        
        debug_dict = defaultdict(dict) if debug else None

        for c in range(self.C):
            if c not in labels:
                continue
            
            class_fg_mask = patch_labels_flat == c  # shape: [N,]
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
        
        if debug:
            return dict(debug_dict)

        return None
                

    def get_pseudo_patch_labels(self, patch_tokens: torch.Tensor, labels: torch.Tensor):
        B, n_patches, dim = patch_tokens.shape
        H = W = int(sqrt(n_patches))
        U, _, _ = torch.pca_lowrank(patch_tokens.reshape(-1, self.dim),
                                    q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_patch_labels = torch.where(U_scaled < self.pca_fg_threshold,
                                          repeat(labels, "B -> B H W", H=H, W=W),
                                          self.C - 1)
        
        return pseudo_patch_labels.to(dtype=torch.long)  # shape: [B, H, W,]

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *, debug: bool = False, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)
        
        patch_tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
        
        if labels is not None:
            pseudo_patch_labels = self.get_pseudo_patch_labels(patch_tokens.detach(), labels=labels)
            debug_dict = self.update_prototypes(
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
        
        if self.cls_head == "fc":
            image_prototype_logits_flat = rearrange(image_prototype_logits[:, :-1, :],
                                                    "B n_classes K -> B (n_classes K)")
            class_logits = self.fc(image_prototype_logits_flat.detach())  # shape: [B, n_classes,]
        elif self.cls_head == "mean":
            class_logits = image_prototype_logits[:, :-1, :].mean(-1)  # shape: [B, C,]
        else:
            class_logits = image_prototype_logits[:, :-1, :].sum(-1)  # shape: [B, C,]

        outputs =  dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits  # shape: [B, n_classes,]
        )

        if labels is not None:
            outputs["pseudo_patch_labels"] = pseudo_patch_labels
            if debug:
                outputs.update(debug_dict)

        return outputs
