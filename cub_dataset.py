from typing import Callable, Optional

import numpy as np
import torch
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
        self.attributes = torch.tensor(np.loadtxt(attribute_label_path))

    def __getitem__(self, index: int):
        im_path, label = self.samples[index]
        im_pt = self.transform(Image.open(im_path).convert("RGB"))
        return im_pt, label, self.attributes[label, :]
