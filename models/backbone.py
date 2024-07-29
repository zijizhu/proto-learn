from torch import nn


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
