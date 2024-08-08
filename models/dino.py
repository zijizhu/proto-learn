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

    def __init__(self):
        super().__init__()
        self.gamma = 0.9
        self.n_prototypes = 5
        self.n_classes = 200 + 1
        self.pca_fg_threshold = 0.5
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        self.dim = 768
        self.register_buffer("prototypes", torch.zeros(self.n_classes, self.n_prototypes, self.dim))

        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def update_prototypes(self,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          labels: torch.Tensor,
                          patch_labels: torch.Tensor,
                          debug: bool = False):
        patch_labels_flat = patch_labels.flatten()
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")
        
        P_old = self.prototypes.clone()
        
        debug_dict = defaultdict(dict) if debug else None

        for c in range(self.n_classes):
            if c not in labels:
                continue
            
            class_fg_mask = patch_labels_flat == c
            I_c = patches_flat[class_fg_mask]  # shape: [N, dim]
            L_c = L[class_fg_mask, c, :]  # shape: [N, K,]
            
            L_c_assignment, _ = sinkhorn_knopp(L_c)  # shape: [N, K,]
            
            P_c_new = torch.mm(L_c_assignment.t(), I_c)  # shape: [K, dim]
            
            P_c_old = P_old[c, :, :]
            
            if self.training:
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
                                          self.n_classes - 1)
        
        return pseudo_patch_labels.to(dtype=torch.long)  # shape: [B, H, W,]

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, debug: bool = False):
        patch_tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)
        patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
        
        if labels is not None:
            pseudo_patch_labels = self.get_pseudo_patch_labels(patch_tokens, labels=labels)
            debug_dict = self.update_prototypes(patch_tokens=patch_tokens_norm,
                                                patch_prototype_logits=patch_prototype_logits,
                                                labels=labels,
                                                patch_labels=pseudo_patch_labels,
                                                debug=debug)
        
        image_prototype_logits, _ = patch_prototype_logits.max(1)  # shape: [B, C, K,]
        class_logits = image_prototype_logits.sum(-1)  # shape: [B, C,]

        outputs =  dict(
            patch_prototype_logits=patch_prototype_logits,
            image_prototype_logits=image_prototype_logits,
            class_logits=class_logits
        )

        if debug and labels is not None:
            debug_dict["pseudo_patch_labels"] = pseudo_patch_labels
            outputs.update(debug_dict)

        return outputs
