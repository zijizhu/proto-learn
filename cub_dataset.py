from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import ImageFolder


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
        return im_pt, label, self.attributes[label, :]
