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

    def __init__(self, proj: nn.Module, *, gamma: float = 0.9, n_prototypes: int = 5, n_classes: int = 200,
                 pca_fg_threshold: float = 0.5, backbone_dim: int = 768, dim: int = 256):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes + 1
        self.pca_fg_threshold = pca_fg_threshold
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        self.backbone_dim = backbone_dim
        self.proj = proj
        self.dim = dim
        self.register_buffer("prototypes", torch.zeros(self.n_classes, self.n_prototypes, self.dim))

        nn.init.trunc_normal_(self.prototypes, std=0.02)

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
        
        patch_prototype_indices = torch.full_like(patch_labels_flat, self.n_classes - 1, dtype=torch.long)  # shape: [B*H*W,]
        
        debug_dict = defaultdict(dict) if debug else None

        for c in range(self.n_classes):
            if c not in labels:
                continue
            
            class_fg_mask = patch_labels_flat == c  # shape: [N,]
            I_c = patches_flat[class_fg_mask]  # shape: [N, dim]
            L_c = L[class_fg_mask, c, :]  # shape: [N, K,]
            
            L_c_assignment, L_c_assignment_indices = sinkhorn_knopp(L_c, use_gumbel=use_gumbel)  # shape: [N, K,], [N,]
            
            P_c_new = torch.mm(L_c_assignment.t(), I_c)  # shape: [K, dim]
            
            P_c_old = P_old[c, :, :]
            
            if self.training:
                self.prototypes[c, ...] = momentum_update(P_c_old, P_c_new, momentum=self.gamma)
            
            if debug:
                debug_dict["L_c"][c] = L_c
                debug_dict["L_c_assignment"][c] = L_c_assignment
            
            patch_prototype_indices[class_fg_mask] = L_c_assignment_indices + (c * self.n_prototypes)
        
        if debug:
            return patch_prototype_indices, dict(debug_dict)

        return patch_prototype_indices, None  # shape: [N,]
                

    def get_pseudo_patch_labels(self, patch_tokens: torch.Tensor, labels: torch.Tensor):
        B, n_patches, dim = patch_tokens.shape
        H = W = int(sqrt(n_patches))
        U, _, _ = torch.pca_lowrank(patch_tokens.reshape(-1, self.backbone_dim),
                                    q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_patch_labels = torch.where(U_scaled < self.pca_fg_threshold,
                                          repeat(labels, "B -> B H W", H=H, W=W),
                                          self.n_classes - 1)
        
        return pseudo_patch_labels.to(dtype=torch.long)  # shape: [B, H, W,]

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, *, debug: bool = False, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)
        
        backbone_patch_tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]  # shape: [B, n_pathes, dim,]
        
        patch_tokens = self.proj(backbone_patch_tokens)  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
        
        if labels is not None:
            pseudo_patch_labels = self.get_pseudo_patch_labels(backbone_patch_tokens.detach(), labels=labels)
            patch_prototype_indices, debug_dict = self.update_prototypes(patch_tokens=patch_tokens_norm.detach(),
                                                patch_prototype_logits=patch_prototype_logits.detach(),
                                                labels=labels.detach(),
                                                patch_labels=pseudo_patch_labels.detach(),
                                                debug=debug,
                                                use_gumbel=use_gumbel)
        
        image_prototype_logits, _ = patch_prototype_logits.max(1)  # shape: [B, C, K,], C=n_classes+1
        class_logits = image_prototype_logits.sum(-1)  # shape: [B, C,]

        outputs =  dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits[:, :-1]  # shape: [B, n_classes,]
        )
        
        if labels is not None:
            outputs["pseudo_patch_labels"] = pseudo_patch_labels  # shape: [B, H, W,]
            outputs["patch_prototype_indices"] = patch_prototype_indices  # shape: [B*H*W,]
            if debug:
                outputs.update(debug_dict)

        return outputs


class Losses(nn.Module):
    """Four Losses:
    Pixel-level cross-entropy (segmentation-like) with pseudo labels;
    Pixel-prototype contrastive loss;
    Standard image-level cross-entropy;
    Cluster-separation loss from ProtoPNet."""
    def __init__(self, l_patch_xe_coef: float, l_contrast_coef: float, l_im_xe_coef: float, l_clst_ceof: float, l_sep_coef: float,
                 patch_xe_ignore_index: int = 200, contrast_ignore_bg: bool = False, n_classes = 200, n_prototypes = 5) -> None:
        super().__init__()
        self.l_patch_xe_coef = l_patch_xe_coef
        self.l_contrast_coef = l_contrast_coef
        self.l_im_xe_coef = l_im_xe_coef
        self.l_clst_ceof = l_clst_ceof
        self.l_sep_coef = l_sep_coef
        
        self.contrast_ignore_bg = contrast_ignore_bg
        self.n_classes, self.n_prototypes = n_classes, n_prototypes
        
        self.im_xe = nn.CrossEntropyLoss()
        # self.patch_xe = nn.CrossEntropyLoss(ignore_index=patch_xe_ignore_index)
        self.patch_xe = nn.CrossEntropyLoss()
        self.patch_prototype_contrast = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict[str, torch.Tensor], image_labels):
        B, H, W = outputs["pseudo_patch_labels"].shape
        loss_dict = dict()
        if self.l_im_xe_coef != 0:
            loss_dict["l_im_xe"] = self.l_im_xe_coef * self.im_xe(outputs["class_logits"], image_labels)
        if self.l_patch_xe_coef != 0:
            patch_class_logits = rearrange(outputs["patch_prototype_logits"].sum(-1), "B (H W) C -> B C H W", H=H, W=W)
            # print(patch_class_logits.shape, outputs["pseudo_patch_labels"].shape, outputs["pseudo_patch_labels"].max())
            loss_dict["l_patch_xe"] = self.l_patch_xe_coef * self.patch_xe(patch_class_logits,
                                                                           outputs["pseudo_patch_labels"])
            # print(loss_dict)
        if self.l_contrast_coef != 0:
            patch_prototype_labels = outputs["patch_prototype_indices"]
            mask = patch_prototype_labels < (self.n_classes * self.n_prototypes)

            contrast_logits = rearrange(outputs["patch_prototype_logits"], "B (H W) C K -> (B H W) (C K)", H=H, W=W)[mask, :-5]
            contrast_labels = patch_prototype_labels[mask]
            # print(contrast_logits.shape, contrast_labels.shape, contrast_labels.max(), contrast_labels.min())
            loss_dict["l_contrast"] = self.l_contrast_coef * self.patch_prototype_contrast(contrast_logits, contrast_labels)
        # print(loss_dict)
        # TODO add computations for cluster and sep loss if necessary
        
        return loss_dict
