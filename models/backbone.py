from functools import partial
from math import sqrt

import torch
from einops import rearrange
from torch import nn

from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import DinoVisionTransformer

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
        self.dino = torch.hub.load("facebookresearch/dinov2", name)
        self.dim = DIM_DICT[name]

    def forward(self, x: torch.Tensor, key: str = "x_norm_patchtokens", reshape: bool = False) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        
        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature


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
            for name, param in self.dino.named_parameters():
                param.requires_grad = name in learnable_param_names
        else:
            self.dino = torch.hub.load('facebookresearch/dinov2', name[:-1])  # type: nn.Module
            self.learnable_param_names = []
    
    def _check(self):
        print("Learnable parameters:")
        for name, param in self.dino.named_parameters():
            if param.requires_grad:
                print(name, "(zero-ed)" if param.detach().sum() == 0 else "")
    
    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names
        self._check()

    def forward(self, x: torch.Tensor, key: str = "x_norm_patchtokens", reshape: bool = False) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        
        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature


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
