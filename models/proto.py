from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from torch import nn

from models.utils import (
    distributed_sinkhorn,
    l2_normalize,
    momentum_update,
)


class ProtoNet(nn.Module):
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
        self.pseudo_gt_threshold = 0.5
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        self.backbone_dim = 768
        self.dim = 256
        self.proj = nn.Linear(self.backbone_dim, self.dim)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.num_prototype, self.dim))

        self.feat_norm = nn.LayerNorm(self.dim)
        self.class_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def update_prototypes(self, patches, patch_class_logits, pseudo_gt, masks, debug=False):
        """
        patches: l2-normalized features of the whole batch, shape: [(B*H*W), dim]
        patch_class_logits: batch class logits for each patch, shape: [B, C, H, W]
        pseudo_gt: Pseudo ground truth class id for each class, shape: [(B*H*W),]
        patch_prototype_logits: batch prototype logits for each patch, shape: [(B*H*W), c, C]
        """
        _, preds = torch.max(patch_class_logits, 1)  # patch prediciton in int64, shape: [B, H, W]
        # flat bool arrary indicate if prediction in batch is correct, shape: [(B*H*W),]
        correct = (pseudo_gt == preds.view(-1))

        # cosine similarity of l2 normalized batch features and prototypes shape: [(B*H*W), (C*c)]
        cosine_similarity = torch.mm(patches, self.prototypes.view(-1, self.dim).t())

        proto_logits = cosine_similarity  # shape: shape: [(B*H*W), (c*m)]
        proto_target = pseudo_gt.clone().float()  # shape: [b, h, w]

        # Perform clustering for each class
        # And update prototypes with weighted mean of the clusters
        protos = self.prototypes.data.clone()
        if debug:
            q_dict = dict()
        else:
            q_dict = None
        for c in range(self.num_classes):
            init_q = masks[..., c]  # shape: [(B*H*W), m]
            init_q = init_q[pseudo_gt == c, ...]  # shape: [n, m]
            if init_q.shape[0] == 0:
                continue

            L, indexs = distributed_sinkhorn(init_q)

            correct_c = correct[pseudo_gt == c]  # shape: [n,], dtype: bool

            patches_c = patches[pseudo_gt == c, ...]  # shape: [n, dim]

            L_correct_mask = repeat(correct_c, 'n -> n dim', dim=self.num_prototype)  # shape: [n, m], dtype: bool

            # Final prototype patch_prototype_logits for each pixel masked by if they are correctly clustered
            L_correct = L * L_correct_mask  # N x K

            patch_correct_mask = repeat(correct_c, 'n -> n dim', dim=self.dim)
            
            # Features masked by whether they are correctly clustered
            patches_c_correct = patches_c * patch_correct_mask  # n x embedding_dim

            f = L_correct.transpose(0, 1) @ patches_c_correct  # K x dim

            n = torch.sum(L_correct, dim=0) # shape: [,self.num_prototype]

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[c, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[c, n != 0, :] = new_value

            proto_target[pseudo_gt == c] = indexs.float() + (self.num_prototype * c)
            
            if debug:
                q_dict[c] = L.detach()

        self.prototypes = l2_normalize(protos).detach().clone()
        return proto_logits, proto_target, q_dict

    def get_pseudo_gt(self, patch_tokens: torch.Tensor, labels: torch.Tensor):
        B, HW, C = patch_tokens.shape
        H = W = int(sqrt(HW))
        U,_, _ = torch.pca_lowrank(patch_tokens.reshape(-1, self.backbone_dim),
                                   q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_gt = torch.where(U_scaled < self.pseudo_gt_threshold,
                                repeat(labels, "b -> b H W", H=H, W=W),
                                self.num_classes - 1)
        
        return pseudo_gt

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, debug: bool = False):
        feature_dict = self.backbone.forward_features(x)
        patch_tokens = feature_dict["x_norm_patchtokens"]
        if labels is not None:
            pseudo_gt = self.get_pseudo_gt(patch_tokens.detach(), labels)
        f = self.proj(patch_tokens)

        B, HW, _ = f.shape
        H = W = int(sqrt(HW))

        patches = rearrange(f, 'B HW dim -> (B HW) dim')
        patches = self.feat_norm(patches)
        patches = l2_normalize(patches)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # N: B*H*W, C: num_class, c: num_prototype
        patch_prototype_logits = torch.einsum('ND,CKD->NKC', patches, self.prototypes)

        patch_class_logits = torch.amax(patch_prototype_logits, dim=1)
        patch_class_logits = self.class_norm(patch_class_logits)
        patch_class_logits = rearrange(patch_class_logits, "(B H W) C -> B C H W", B=B, H=H, W=W)

        if labels is not None:
            pseudo_gt = pseudo_gt.reshape(-1)
            contrast_logits, contrast_target, q_dict = self.update_prototypes(patches, patch_class_logits, pseudo_gt.reshape(-1), patch_prototype_logits, debug=debug)
            return {'seg': patch_class_logits, 'patch_prototype_logits': contrast_logits, "prototype_logits": patch_prototype_logits,
                    'target': contrast_target, "pseudo_gt": pseudo_gt, "q_dict": q_dict}

        return {"class_logits": patch_class_logits, "prototype_logits": patch_prototype_logits}