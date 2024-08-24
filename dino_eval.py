from math import sqrt
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from einops import rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from cub_dataset import CUBEvalDataset


def in_bbox(keypoint: tuple[float, float], bbox: tuple[float, float, float, float]):
    kp_x, kp_y = keypoint
    x, y, w, h = bbox
    return (x <= kp_x <= x + w) and (y <= kp_y <= y + h)


def compute_part_vectors(activation_maps: torch.Tensor, keypoints: torch.Tensor, height=72, width=72):
    kp_visibility = keypoints.sum(dim=-1).to(dtype=torch.bool).to(dtype=torch.long)
    part_vectors = []
    K, H, W = activation_maps.shape
    for act_map in activation_maps:
        cy, cx = torch.unravel_index(torch.argmax(act_map), (H, W,))
        bbox_coord = (max(cx - width // 2, 0), max(cy - height //2, 0), width, height,)
        kp_in_part = torch.tensor([in_bbox(kp, bbox_coord) for kp in keypoints], dtype=torch.long)
        part_vectors.append(kp_in_part)
        
    return torch.stack(part_vectors), repeat(kp_visibility, "n_parts -> K n_parts", K=K)


def eval_consistency(net: nn.Module, dataloader: DataLoader,
                    *,
                    K: int = 5, C: int = 200, N_PARTS: int = 15,
                    H_b: int = 72, W_b: int = 72, INPUT_SIZE: int = 224, device: str | torch.device = "cpu"):

    O_p, O_sum = torch.zeros((C, K, N_PARTS,), dtype=torch.long), torch.zeros((C, K, N_PARTS,), dtype=torch.long)

    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)
        transformed_im, transformed_keypoints, labels, attributes, _ = batch
        outputs = net(transformed_im)

        for activations, keypoints, c in zip(outputs["patch_prototype_logits"], transformed_keypoints, labels):
            H = W = int(sqrt(activations.size(0)))
            activations = rearrange(activations, "(H W) C K -> C K H W", H=H, W=W)
            activations = F.interpolate(activations, INPUT_SIZE, mode="bicubic")  # shape: [C, K, INPUT, INPUT,]
            activations_c = activations[c, ...]  # shape: [K, H, W,]
            part_vectors, part_visibilities = compute_part_vectors(activation_maps=activations_c, keypoints=keypoints, height=H_b, width=W_b)  # shape: [K, N_PARTS]

            O_p[c] += part_vectors
            O_sum[c] += part_visibilities

    a_p = O_p / O_sum
    return torch.mean(a_p.max(-1).values >= 0.8)


if __name__ == "__main__":
    dataset_dir = Path("datasets") / "cub200_cropped"
    annotations_path = Path("datasets") / "CUB_200_2011"

    dataset_eval = CUBEvalDataset((dataset_dir / "test_cropped").as_posix(), annotations_path.as_posix())

    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=128)
    dataloader_eval_iter = iter(dataloader_eval)
    
    net = ...
    device = torch.device
    
    net.eval()
    net.to(device)
    
    