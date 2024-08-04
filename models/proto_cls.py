from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from models.utils import (
    distributed_sinkhorn,
    l2_normalize,
    momentum_update,
)


class ProtoNetCLS(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self):
        super().__init__()
        self.gamma = 0.999
        self.num_prototype = 5
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False
        self.num_classes = 200 + 1
        self.pseudo_fg_mask_threshold = 0.5
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        self.backbone_dim = 768
        self.dim = 256
        self.proj = nn.Linear(self.backbone_dim, self.dim)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.num_prototype, self.dim))

        self.feat_norm = nn.LayerNorm(self.dim)
        self.class_norm = nn.LayerNorm(self.num_classes)
        
        self.max_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1,))
        fc = torch.eye(self.num_classes, dtype=torch.float32).repeat_interleave(self.num_prototype, dim=0)
        self.register_buffer("fc", fc)

        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def update_prototypes(self, patches, patch_class_logits, pseudo_fg_mask, patch_prototype_logits, topk=1, debug=False):
        """
        patches: l2-normalized features of the whole batch, shape: [(B*H*W), dim]
        patch_class_logits: batch class logits for each patch, shape: [B, C, H, W]
        pseudo_fg_mask: class/background id for each patch, shape: [(B*H*W),]
        patch_prototype_logits: batch prototype logits for each patch, shape: [(B*H*W), C, K]
        """
        patch_class_preds = torch.argmax(patch_class_logits, dim=1)  # patch prediciton in int64, shape: [B, H, W]
        # flat bool arrary indicate if prediction in batch is correct, shape: [(B*H*W),]
        correct = (pseudo_fg_mask == patch_class_preds.view(-1))

        # cosine similarity of l2 normalized batch features and prototypes shape: [(B*H*W), (C*K)]
        cosine_similarities = torch.mm(patches, self.prototypes.view(-1, self.dim).t())
        cos = rearrange(cosine_similarities, "N (C K) -> N C K", C=self.num_classes, K=self.num_prototype)

        contrast_targets = pseudo_fg_mask.clone().float()  # shape: [b, h, w]

        # Perform clustering for each class
        # And update prototypes with weighted mean of the clusters
        P = self.prototypes.detach().clone()

        L_dict = dict() if debug else None
        n_dict = dict() if debug else None
        n_final_dict = dict() if debug else None

        for c in range(self.num_classes):
            L_init = patch_prototype_logits[:, c, :]  # shape: [(B*H*W), K]
            L_init = L_init[pseudo_fg_mask == c, ...]  # shape: [N, K]
            if L_init.shape[0] == 0:
                continue

            L, prototype_indices = distributed_sinkhorn(L_init)

            correct_c = correct[pseudo_fg_mask == c]  # shape: [n,], dtype: bool

            patches_c = patches[pseudo_fg_mask == c, ...]  # shape: [n, dim]

            L_correct_mask = repeat(correct_c, 'n -> n dim', dim=self.num_prototype)  # shape: [n, m], dtype: bool

            # Pixel-prototype assignement matrix masked by if they are correctly predicted
            L_correct = L * L_correct_mask  # N x K
            cos_L_correct = cos[:, c, :] * L_correct
            _, topk_patch_indices = cos_L_correct.topk(k=1, dim=0)
            final_assignment = torch.zeros_like(cos_L_correct, dtype=torch.float32)
            final_assignment[topk_patch_indices, range(self.num_prototype)] = 1.

            patches_prototype_logits = repeat(correct_c, 'n -> n dim', dim=self.dim)
            
            # Features masked by whether they are correctly predicted
            patches_c_correct = patches_c * patches_prototype_logits  # n x embedding_dim

            # Batch prototype mean
            P_new = final_assignment.transpose(0, 1) @ patches_c_correct  #  shape: [K, dim]

            n = torch.sum(L_correct, dim=0) # shape: [,self.num_prototype]
            n_final = torch.sum(final_assignment, dim=0) # shape: [,self.num_prototype]

            if torch.sum(n_final) > 0 and self.update_prototype is True:
                P_new = F.normalize(P_new, p=2, dim=-1)

                P_c_updated = momentum_update(old_value=P[c, n_final != 0, :],
                                              new_value=P_new[n_final != 0, :],
                                              momentum=self.gamma,
                                              debug=False)
                P[c, n_final != 0, :] = P_c_updated

            contrast_targets[pseudo_fg_mask == c] = prototype_indices.float() + (self.num_prototype * c)
            
            if debug:
                L_dict[c] = L.detach()
                n_dict[c] = n
                n_final_dict[c] = n_final

        self.prototypes = l2_normalize(P).detach().clone()
        return cosine_similarities, contrast_targets, L_dict, n_dict, n_final_dict

    def get_pseudo_fg_mask(self, patch_tokens: torch.Tensor, labels: torch.Tensor):
        B, HW, C = patch_tokens.shape
        H = W = int(sqrt(HW))
        U,_, _ = torch.pca_lowrank(patch_tokens.reshape(-1, self.backbone_dim),
                                   q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_fg_mask = torch.where(U_scaled < self.pseudo_fg_mask_threshold,
                                repeat(labels, "b -> b H W", H=H, W=W),
                                self.num_classes - 1)
        
        return pseudo_fg_mask

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, debug: bool = False):
        feature_dict = self.backbone.forward_features(x)
        patch_tokens = feature_dict["x_norm_patchtokens"]
        if labels is not None:
            pseudo_fg_mask = self.get_pseudo_fg_mask(patch_tokens.detach(), labels)
        patches = self.proj(patch_tokens)

        B, HW, _ = patches.shape
        H = W = int(sqrt(HW))

        patches = rearrange(patches, 'B HW dim -> (B HW) dim')
        patches = self.feat_norm(patches)
        patches = l2_normalize(patches)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # N: B*H*W, C: num_class, c: num_prototype
        patch_prototype_logits = torch.einsum('ND,CKD->NCK', patches, self.prototypes)  # range: [0, 1]
        
        patch_prototype_logits = rearrange(patch_prototype_logits, "(B H W) C K -> B (C K) H W", B=B, H=H, W=W)
        image_prototype_logits = self.max_pool(patch_prototype_logits).squeeze()  # type: torch.Tensor  # shape: [B, C*K,]
        
        pred_logits = image_prototype_logits @ self.fc
        

        if labels is not None:
            contrast_logits, contrast_target, L_dict, n_dict, n_final_dict = self.update_prototypes(
                patches,
                patch_prototype_logits.amax(dim=-1),
                pseudo_fg_mask.reshape(-1),
                patch_prototype_logits,
                debug=debug
            )
            return dict(
                pred_logits=pred_logits,
                patch_prototype_logits=patch_prototype_logits,
                image_prototype_logits=image_prototype_logits,
                contrast_logits=contrast_logits,
                contrast_target=contrast_target,
                pseudo_fg_mask=pseudo_fg_mask,
                # outputs for debugging
                L_dict=L_dict,
                n_dict=n_dict,
                n_final_dict=n_final_dict
            )
        return dict(
                pred_logits=pred_logits,
                patch_prototype_logits=patch_prototype_logits
            )


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.xe_coef = 1.
        self.clst_coef = 0.8
        self.sep_coef = 0.08
        self.contrast_coef = 0.01
        
        self.xe = nn.CrossEntropyLoss()
        
    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, proto_class_association: torch.Tensor):
        loss_dict = dict()
        xe = self.xe(outputs["pred_logits"], labels)
        loss_dict["xe"] = self.xe_coef * xe
        
        if self.clst_coef > 0 and self.sep_coef > 0 and outputs.get("image_prototype_logits", None):
            similarities = outputs["image_prototype_logits"]
            gt_proto_mask = proto_class_association[:, labels].T
            gt_proto_inverted_dists, _ = torch.max(similarities * gt_proto_mask, dim=1)
            cluster_cost = torch.mean(1 - gt_proto_inverted_dists)
            loss_dict["clst"] = self.clst_coef * cluster_cost
            
            non_gt_proto_mask = 1 - gt_proto_mask
            non_gt_proto_inverted_dists, _ = torch.max(similarities * non_gt_proto_mask, dim=1)
            separation_cost = torch.mean(1 - non_gt_proto_inverted_dists)
            loss_dict["sep"] = self.sep_coef * separation_cost
        
        return loss_dict