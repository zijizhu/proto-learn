import pickle as pkl
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.augmentations.crops.functional import crop_keypoint_by_coords
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor


class CUBDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            transform=transforms,
            target_transform=target_transform
        )

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        return im_pt, label, index


class CUBConceptDataset(ImageFolder):
    def __init__(self,
                 image_root: str,
                 attribute_annotation_root: str,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=image_root,
            transform=transforms,
            target_transform=target_transform
        )

        with open(Path(attribute_annotation_root) / "train.pkl", "rb") as fp:
            train_attribute_anns = pkl.load(fp)

        label2attr = dict()
        for ann in train_attribute_anns:
            label, attribute_vector = ann["class_label"], ann["attribute_label"]
            if label not in label2attr:
                label2attr[label] = attribute_vector

        self.attributes = torch.tensor([label2attr[i] for i in range(len(label2attr))], dtype=torch.long)

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        attr = self.attributes[label]
        return im_pt, label, attr, index


class CUBEvalDataset(ImageFolder):
    def __init__(self,
                 images_root: str,
                 annotations_root: str,
                 normalization: bool = True,
                 input_size: int = 224):
        transforms = [A.Resize(width=input_size, height=input_size)]
        transforms += [A.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))] if normalization else []

        super().__init__(
            root=images_root,
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
        self.input_size = input_size
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
        keypoints_cropped = [(np.clip(x, 0, self.input_size), np.clip(y, 0, self.input_size),) for x, y, _, _ in keypoints_cropped]
        
        transformed = self.transform(image=im, keypoints=keypoints_cropped)
        transformed_im, transformed_keypoints = transformed["image"], transformed["keypoints"]
        
        return to_tensor(transformed_im), torch.tensor(transformed_keypoints, dtype=torch.float32), label, self.attributes[label, :], index


class CUBFewShotDataset(ImageFolder):
    def __init__(self,
                 root: str,
                 n_samples_per_class: int,
                 transforms: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(
            root=root,
            transform=transforms,
            target_transform=target_transform
        )
        self.n_samples_per_class = n_samples_per_class
        label_to_paths = defaultdict(list)
        for path, label in self.samples:
            label_to_paths[label].append(path)

        self.samples = []
        for label, class_sampels in label_to_paths.items():
            self.samples += [(sample, label,) for sample in random.sample(class_sampels, k=n_samples_per_class)]

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        return im_pt, label, index
