from typing import Callable, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor


class CUBDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 attribute_label_path: str,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            transform=transforms,
            target_transform=target_transform
        )
        attributes_np = np.loadtxt(attribute_label_path)
        self.attributes = F.normalize(torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        return im_pt, label, self.attributes[label, :], index


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 annotations_root: str,
                 normalization: bool = True,
                 input_size: int = 224):
        transforms = [A.Resize(width=input_size, height=input_size)]
        transforms += [A.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))] if normalization else []

        super().__init__(
            root=root,
            transform=A.Compose(
                transforms,
                keypoint_params=A.KeypointParams(
                    format='xy',
                    label_fields=None,
                    remove_invisible=True,
                    angle_in_degrees=True
                )
            ),
            target_transform=None
        )
        annotations_root = Path("datasets") / "CUB_200_2011"

        path_df = pd.read_csv(annotations_root / "images.txt", header=None, names=["image_id", "image_path"], sep=" ")
        bbox_df = pd.read_csv(annotations_root / "bounding_boxes.txt", header=None, names=["image_id", "x", "y", "w", "h"], sep=" ")
        self.bbox_df = path_df.merge(bbox_df, on="image_id")
        self.part_loc_df = pd.read_csv(annotations_root / "parts" / "part_locs.txt", header=None, names=["image_id", "part_id", "kp_x", "kp_y", "visible"], sep=" ")
        
        attributes_np = np.loadtxt(annotations_root / "attributes" / "class_attribute_labels_continuous.txt")
        self.attributes = F.normalize(torch.tensor(attributes_np, dtype=torch.float32), p=2, dim=-1)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im = np.array(Image.open(im_path).convert("RGB"))

        row = self.bbox_df[self.bbox_df["image_path"] == "/".join(Path(im_path).parts[-2:])].iloc[0]
        image_id = row["image_id"]
        bbox_coords = row[["x", "y", "w", "h"]].values.flatten()
        
        mask = self.part_loc_df["image_id"] == image_id
        keypoints = self.part_loc_df[mask][["kp_x", "kp_y"]].values

        keypoints_cropped = [crop_keypoint_by_coords(keypoint=tuple(kp) + (None, None,), crop_coords=bbox_coords[:2]) for kp in keypoints]
        keypoints_cropped = [(max(x, 0), max(y, 0),) for x, y, _, _ in keypoints_cropped]
        
        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = transformed["image"], transformed["keypoints"]
        
        return to_tensor(transformed_im), np.array(transformed_keypoints), label, self.attributes[label, :], index


"""
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

K = 5
C = 200
N_PARTS = 15
H_b = W_b = 72
INPUT_SIZE = 224

net = nn.Module()

O_p, O_sum = torch.zeros((C, K, N_PARTS,)), torch.zeros((C, K, N_PARTS,))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch in dataloader_eval_iter:
    batch = tuple(item.to(device) for item in batch)
    transformed_im, transformed_keypoints, labels, attributes, _ = batch
    outputs = net(transformed_im)

    for activations, keypoints, c in zip(outputs["patch_prototype_logits"], transformed_keypoints, labels):
        activations = F.interpolate(activations, INPUT_SIZE, mode="bicubic")  # shape: [C, K, H, W,]
        activations_c = activations[c, ...]  # shape: [K, H, W,]
        part_vectors, part_visibilities = compute_part_vectors(activation_maps=activations_c, keypoints=keypoints)
        O_p[c] += part_vectors
        O_sum[c] += part_visibilities

a_p = O_p / O_sum
torch.mean(a_p.max(-1).values >= 0.8)
"""
