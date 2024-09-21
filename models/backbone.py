import os
import re
from functools import partial
from math import sqrt

import torch
from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import DinoVisionTransformer
from einops import rearrange
from torch import nn
# from mmpretrain import get_model

from .maskclip import clip
from .utils import block_expansion_dino

common_kwargs = dict(
    img_size=518,
    patch_size=14,
    mlp_ratio=4,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_fn=partial(Block, attn_class=MemEffAttention)
)

vit_small_kwargs = dict(embed_dim=384, num_heads=6)
vit_base_kwargs = dict(embed_dim=768, num_heads=12)

MODEL_DICT = {
    "dinov2_vits14_reg4": partial(DinoVisionTransformer, **vit_small_kwargs, **common_kwargs),
    "dinov2_vitb14_reg4": partial(DinoVisionTransformer, **vit_base_kwargs, **common_kwargs)
}

URL_DICT = {
    "dinov2_vits14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
}

DIM_DICT = {
    "dinov2_vits14_reg4": 384,
    "dinov2_vitb14_reg4": 768
}


class DINOv2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14_reg"):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", name[:-1])  # type: nn.Module
        self.dim = DIM_DICT[name]
    
    def learnable_parameters(self):
        return self.dino.parameters()
    
    def set_requires_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, key: str = "x_norm_patchtokens", cls_key: str = "x_norm_clstoken", reshape: bool = False) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        cls_token = feature_dict[cls_key]
        
        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature, cls_token


class DINOv2BackboneExpanded(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14_reg", n_splits: int = 0):
        super().__init__()
        self.dim = DIM_DICT[name]
        if n_splits > 0:
            arch = MODEL_DICT[name]
            state_dict = torch.hub.load_state_dict_from_url(URL_DICT[name], map_location="cpu")
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = block_expansion_dino(
                state_dict=state_dict,
                n_splits=n_splits)
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = torch.hub.load('facebookresearch/dinov2', name[:-1])  # type: nn.Module
            self.learnable_param_names = []
    
    def learnable_parameters(self):
        return list(param for name, param in self.dino.named_parameters() if name in self.learnable_param_names)
    
    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def forward(self,
                x: torch.Tensor,
                feature_key: str = "x_norm_patchtokens",
                cls_key: str = "x_norm_clstoken",
                reshape_features: bool = False) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[feature_key]
        cls_token = feature_dict[cls_key]
        
        B, n_patches, dim = feature.shape

        if reshape_features:
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature, cls_token


class MaskCLIP(nn.Module):
    """
    Implementation adapted from https://github.com/mhamilton723/FeatUp/tree/main/featup/featurizers
    """
    def __init__(self, name: str = "ViT-B/16"):
        super().__init__()
        self.model, self.preprocess = clip.load(
            name,
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features


# class MOCO(nn.Module):
#     def __init__(self, name: str = "mocov3_resnet50_8xb512-amp-coslr-800e_in1k") -> None:
#         super().__init__()

#         self.model = get_model(
#             name,
#             pretrained=True,
#             data_preprocessor=dict(
#                 type='SelfSupDataPreprocessor',
#                 mean=(123.6750, 116.2800, 103.5300,),
#                 std=(58.3950, 57.1200, 57.3750,),
#                 to_rgb=True
#             )
#         )
    
#     def forward(self, x):
#         (x,) = self.model(x)
#         return rearrange(x, "B dim H W -> B (H W) dim")

def load_backbone(backbone_name: str) -> tuple[nn.Module, int]:
    if backbone_name == 'resnet50':
        from torchvision.models import ResNet50_Weights, resnet50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        return nn.Sequential(*list(backbone.children())[:-2]), 2048
    if backbone_name == 'resnet34':
        from torchvision.models import ResNet34_Weights, resnet34
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        return nn.Sequential(*list(backbone.children())[:-2]), 512
    elif backbone_name == 'vgg19':
        from torchvision.models import VGG19_Weights, vgg19
        backbone = vgg19(weights=VGG19_Weights.DEFAULT)
        return backbone.features, 512
    elif backbone_name == 'densenet121':
        from torchvision.models import DenseNet121_Weights, densenet121
        backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        return backbone.features, 1024
    else:
        raise NotImplementedError
    

def load_adapter(config: list[int]):
    if not config:
        return nn.Identity()
    pattern = r"^\d+,\d+$"
    layers = []
    for s in config:
        if re.match(pattern=pattern, string=s):
            in_dim, out_dim = s.split(",")
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
        elif s == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError
    return nn.Sequential(*layers)
