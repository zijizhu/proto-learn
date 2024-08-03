import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum
from math import sqrt
from collections import defaultdict

from timm.models.layers import trunc_normal_

from models.utils import momentum_update, distributed_sinkhorn, l2_normalize, PixelPrototypeCELoss


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
        in_channels = 256
        self.proj = nn.Linear(self.backbone_dim, in_channels)
        self.register_buffer("prototypes", torch.zeros(self.num_classes, self.num_prototype, in_channels))

        self.feat_norm = nn.LayerNorm(in_channels)
        self.class_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def update_prototypes(self, _c, out_seg, gt_seg, masks, debug=False):
        """
        _c: l2-normalized features of the whole batch, shape: [(b*h*w), d]
        out_seg: batch segmentation class logits, shape: [b, k, h, w]
        gt_seg: batch segmentation ground truth (downsampled and turned into float), shape: [(b*h*w),]
        masks: batch segmentation logits per prototype, shape: [(b*h*w), m, k]
        """
        _, pred_seg = torch.max(out_seg, 1)  # sengmentation prediciton in int64, shape: [b, h, w]
        # flat bool arrary indicate if prediction in batch is correct, shape: [(b*h*w),]
        mask = (gt_seg == pred_seg.view(-1))

        # cosine similarity of l2 normalized batch features and prototypes shape: [(b*h*w), (k*m)]
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity  # shape: shape: [(b*h*w), (k*m)]
        proto_target = gt_seg.clone().float()  # shape: [b, h, w]

        # Perform clustering for each class
        # And update prototypes with weighted mean of the clusters
        protos = self.prototypes.data.clone()
        if debug:
            q_dict = dict()
        else:
            q_dict = None
        for k in range(self.num_classes):
            init_q = masks[..., k]  # shape: [(b*h*w), m]
            init_q = init_q[gt_seg == k, ...]  # shape: [n, m]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]  # shape: [n,], dtype: bool

            c_k = _c[gt_seg == k, ...]  # shape: [n, d]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)  # shape: [n, m], dtype: bool

            # Final prototype logits for each pixel masked by if they are correctly clustered
            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            
            # Features masked by whether they are correctly clustered
            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0) # shape: [,self.num_prototype]

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
            
            if debug:
                q_dict[k] = q.detach()

        self.prototypes = l2_normalize(protos).detach().clone()
        return proto_logits, proto_target, q_dict

    def get_pseudo_gt(self, batch_patch_tokens: torch.Tensor, batch_labels: torch.Tensor):
        B, HW, C = batch_patch_tokens.shape
        H = W = int(sqrt(HW))
        U,_, _ = torch.pca_lowrank(batch_patch_tokens.reshape(-1, self.backbone_dim),
                                   q=1, center=True, niter=10)
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()
        U_scaled = U_scaled.reshape(B, H, W)
        
        pseudo_gt = torch.where(U_scaled < self.pseudo_gt_threshold,
                                repeat(batch_labels, "b -> b H W", H=H, W=W),
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

        # N: b*h*w, C: num_class, K: num_prototype
        logits = torch.einsum('ND,CKD->NKC', patches, self.prototypes)

        patch_class_assignments = torch.amax(logits, dim=1)
        patch_class_assignments = self.class_norm(patch_class_assignments)
        patch_class_assignments = rearrange(patch_class_assignments, "(B H W) C -> B C H W", B=B, H=H, W=W)

        if labels is not None:
            gt_seg = pseudo_gt.reshape(-1)
            contrast_logits, contrast_target, q_dict = self.update_prototypes(patches, patch_class_assignments, gt_seg, logits, debug=debug)
            return {'seg': patch_class_assignments, 'logits': contrast_logits, "prototype_logits": logits,
                    'target': contrast_target, "pseudo_gt": pseudo_gt, "q_dict": q_dict}

        return {"class_logits": patch_class_assignments, "prototype_logits": logits}