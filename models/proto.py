import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from math import sqrt

from timm.models.layers import trunc_normal_

from models.utils import momentum_update, distributed_sinkhorn, l2_normalize, PixelPrototypeCELoss


class ProtoNet(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self):
        super().__init__()
        self.gamma = 0.999
        self.num_prototype = 10
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False
        self.num_classes = 200
        self.backbone = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14_reg")

        in_channels = 512
        self.proj = nn.Linear(768, 512)
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        """
        _c: l2-normalized features of the whole batch, shape: [(b*h*w), d]
        out_seg: batch segmentation class logits, shape: [b, k, h, w]
        gt_seg: batch segmentation ground truth (downsampled and turned into float), shape: [(b*h*w),]
        masks: batch segmentation logits per prototype, shape: [(b*h*w), m, k]
        """
        _, pred_seg = torch.max(out_seg, 1)  # sengmentation prediciton in int64, shape: [b, h, w]
        # flat bool arrary indicate if prediction in batch is correct, shape: [(b*h*w),]
        mask = (gt_seg == pred_seg.view(-1))
        print("mask:", mask.shape, mask.dtype)

        # cosine similarity of l2 normalized batch features and prototypes shape: [(b*h*w), (k*m)]
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity  # shape: shape: [(b*h*w), (k*m)]
        proto_target = gt_seg.clone().float()  # shape: [b, h, w]

        # Perform clustering for each class
        # And update prototypes with weighted mean of the clusters
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]  # shape: [(b*h*w), m]
            init_q = init_q[gt_seg == k, ...]  # shape: [n, m]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)
            print("q.shape:", q.shape)

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
            print("n.shape:", n.shape)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)
        return proto_logits, proto_target

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        feature_dict = self.backbone.forward_features(x_)
        f = self.proj(feature_dict["x_norm_patchtokens"])
        print(f.shape)

        b, hw, c = f.shape
        h = w = int(sqrt(hw))

        _c = rearrange(f, 'b hw c -> (b hw) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: b*h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
        print(_c.shape)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=b, h=h, w =w)
        print("out_seg.shape:", out_seg.shape, out_seg.dtype)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = gt_semantic_seg.reshape(-1)
            # gt_seg = F.interpolate(gt_semantic_seg.float(), size=(h, w), mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        return out_seg
