from collections import defaultdict
from logging import getLogger
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import Cub2011Eval, mean, std

logger = getLogger(__name__)


def norm_and_thresh(activations: torch.Tensor, threshold: float = 0.7):
    """Converts each activation map to a binary mask by normalizing and thresholding"""
    B, H, H, W = activations.shape
    max_values = F.adaptive_max_pool2d(activations, output_size=(1, 1,))
    min_valies = -F.adaptive_max_pool2d(-activations, output_size=(1, 1,))
    normalized_activations = (activations - min_valies) / (max_values - min_valies)

    binary_activations = (normalized_activations >= threshold).to(dtype=torch.float32)

    return binary_activations


def calculate_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-6):
    intersection = torch.sum(torch.logical_and(mask1, mask2))
    union = torch.sum(torch.logical_or(mask1, mask2))
    iou = intersection / (union + eps)
    return iou


def batch_mean_IoU(batch_activations: torch.Tensor, threshold: float = 0.7):
    B, K, H, W = batch_activations.shape
    iou_matrix_mask = torch.ones((K, K,)).triu(diagonal=1).to(dtype=torch.bool, device=batch_activations.device)

    mean_IoUs = []
    binary_activations = norm_and_thresh(batch_activations, threshold=threshold)
    for activations in binary_activations:
        iou_matrix = torch.zeros((K, K,), device=batch_activations.device)

        for i in range(K):
            for j in range(i + 1, K):
                iou_matrix[i, j] = calculate_iou(activations[i], activations[j])

        iou_matrix_triu = iou_matrix[iou_matrix_mask]
        mean_IoUs.append(iou_matrix_triu.mean().item())

    return mean_IoUs

@torch.no_grad()
def get_attn_maps(outputs: dict[str, torch.Tensor], labels: torch.Tensor):
    patch_prototype_logits = outputs["patch_prototype_logits"]

    batch_size, n_patches, C, K = patch_prototype_logits.shape
    H = W = int(sqrt(n_patches))

    patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
    patch_prototype_logits = patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W

    pooled_logits = F.avg_pool2d(patch_prototype_logits, kernel_size=(2, 2,), stride=2)
    return patch_prototype_logits, pooled_logits

@torch.no_grad()
def evaluate_distinctiveness(net: nn.Module,
                             save_path: str | Path,
                             thresholds: list[float] =  [0.4, 0.5, 0.6, 0.7, 0.8],
                             num_classes: int = 200,
                             device: torch.device = torch.device("cpu"),
                             input_size: tuple[int, int] = (224, 224,)):
    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval("datasets", train=False, transform=transform)    # CUB test dataset
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8, pin_memory=True, drop_last=False, shuffle=True)

    net.to(device)
    net.eval()

    thresh_to_mean_IoUs = defaultdict(list)
    for b, batch in enumerate(tqdm(test_loader)):
        images, targets, img_ids = tuple(item.to(device=device) for item in batch)
        B, _, INPUT_H, INPUT_W = images.shape
        if hasattr(net, 'get_attn_maps'):
            _, batch_activations = net.get_attn_maps(images, targets)
        if hasattr(net, 'push_forward'):
            _, all_batch_activations = net.push_forward(images)
            B, KN, H, W = all_batch_activations.shape
            K = KN // num_classes
            proto_indices = (targets * K).unsqueeze(dim=-1).repeat(1, K)
            proto_indices += torch.arange(K).to(device=device)   # The indexes of prototypes belonging to the ground-truth class of each image
            proto_indices = proto_indices[:, :, None, None].repeat(1, 1, H, W)
            batch_activations = torch.gather(all_batch_activations, 1, proto_indices) # (B, proto_per_class, fea_size, fea_size)
        else:
            outputs = net(images, targets)
            batch_activations = get_attn_maps(outputs, labels=targets)
        batch_activations_resized = F.interpolate(batch_activations, size=(INPUT_H, INPUT_W,), mode="bilinear")

        for thresh in thresholds:
            thresh_to_mean_IoUs[thresh] += batch_mean_IoU(batch_activations_resized, threshold=thresh)
    
    np.savez(Path(save_path) / "threshold_to_mean_IoUs", **{f"{thresh:.1f}": np.array(values) for thresh, values in thresh_to_mean_IoUs.items()})
        
    thresh_to_scores = defaultdict(float)
    for thresh, mean_IoUs in thresh_to_mean_IoUs.items():
        score = 1 - (sum(mean_IoUs) / len(mean_IoUs))
        thresh_to_scores[thresh] = score
        logger.info(f"Distinctiveness Score Threshold {thresh:.1f}: {score:.4f}")
