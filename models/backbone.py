from math import sqrt

import torch
from einops import rearrange
from torch import nn


class DINOv2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14_reg") -> None:
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', name)
    
    def forward(self, x: torch.Tensor, key: str = "x_norm_patch_tokens", reshape: bool = True) -> torch.Tensor:
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
    elif "dinov2" in backbone_name:
        assert backbone_name in ["dinov2_vitb14_reg"]
        name_to_dim = {
            "dinov2_vitb14_reg": 768
        }
        backbone = DINOv2Backbone(name=backbone_name)
        return backbone, name_to_dim[backbone_name]
    else:
        raise NotImplementedError
