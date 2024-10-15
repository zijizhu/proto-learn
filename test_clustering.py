import random
from math import sqrt
from pathlib import Path
from typing import Callable

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import repeat
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from cub_dataset import CUBDataset
from models.backbone import MaskCLIP
from models.utils import sinkhorn_knopp

#### Constants ####
MEAN = (0.485, 0.456, 0.406,)
STD = (0.229, 0.224, 0.225,)
INPUT_SIZE = 224
BATCH_SIZE = 49

N_PROTOTYPES = 5
DIM = 384
GAMMA = 0.99  # coefficient of OLD value

CLASS_ID = 15
N_ITER = BATCH_SIZE
NCOLS = NROWS = int(sqrt(BATCH_SIZE))
#### Constants ####


def get_foreground_by_PCA(patch_tokens: torch.Tensor,
                          labels: torch.Tensor,
                          pca_threshold: float = 0.5,
                          pca_compare_fn: Callable = torch.ge,
                          bg_label: int = 200):
    B, n_patches, dim = patch_tokens.shape
    H = W = int(sqrt(n_patches))
    U, _, _ = torch.pca_lowrank(
        patch_tokens.reshape(-1, dim),
        q=1, center=True, niter=10
    )
    U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()  # shape: [B*H*W, 1]
    U_scaled = U_scaled.reshape(B, H, W)

    pseudo_patch_labels = torch.where(
        pca_compare_fn(U_scaled, pca_threshold),
        repeat(labels, "B -> B H W", H=H, W=W),
        bg_label
    )

    return pseudo_patch_labels.to(dtype=torch.long)  # shape: [B, H, W,]


if __name__ == "__main__":
    L.seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"

    normalize = T.Normalize(mean=MEAN, std=STD)
    transforms = T.Compose([
        T.Resize((INPUT_SIZE, INPUT_SIZE,)),
        T.ToTensor(),
        normalize
    ])

    dataset_train = CUBDataset((dataset_dir / "train_cropped").as_posix(),
                                transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              transforms=transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)

    #### Initialize model ####
    @torch.no_grad()
    def get_features(net: nn.Module, x: torch.tensor):
        features = net.forward_features(x)['x_norm_patchtokens']  # type: torch.Tensor
        return features
    
    net = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14_reg")
    # import timm
    # @torch.no_grad()
    # def get_features(net: nn.Module, x: torch.tensor):
    #     return net(x)
    # # net = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
    # net = MaskCLIP(name="ViT-L/14")
    # net = net.eval()
    #### Initialize model ####

    # class_subset = [(path, label,) for (path, label,) in dataset_train.samples if label == CLASS_ID]
    # class_subset = class_subset[:BATCH_SIZE]

    samples = random.sample(dataset_train.samples, 49)
    images = torch.stack([transforms(Image.open(path)) for (path, label,) in samples])
    labels = torch.full((BATCH_SIZE,), CLASS_ID, dtype=torch.long)


    # images = torch.stack([transforms(Image.open(path)) for (path, label,) in class_subset])
    # labels = torch.full((BATCH_SIZE,), CLASS_ID, dtype=torch.long)

    prototypes = torch.empty((N_PROTOTYPES, DIM,), dtype=torch.float32)
    nn.init.trunc_normal_(prototypes, std=0.02)

    all_features = get_features(net, images)  # shape: [BATCH_SIZE, HW, DIM]
    all_fg_maps = get_foreground_by_PCA(patch_tokens=all_features, labels=labels)  # shape: [BATCH_SIZE, H, W,]
    all_fg_masks = all_fg_maps == CLASS_ID

    fig, axes = plt.subplots(NCOLS, NROWS)
    for i, (patch_tokens, fg_mask, ax) in enumerate(zip(all_features, all_fg_masks, axes.flatten())):
        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        print(patch_tokens_norm.shape)

        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        cos_similarities = torch.mm(patch_tokens_norm, prototypes_norm.t())

        L = cos_similarities[fg_mask.flatten(), :]
        fg_patches_flat_norm = patch_tokens_norm[fg_mask.flatten(), :]

        L_optimized, indices = sinkhorn_knopp(L)  # shape: [N_FG, N_PROTOTYPES,], [N_FG,]

        # Visualization
        L_vis = L_optimized.clone()
        L_argmax = L_vis.argmax(dim=-1)  # shape: [N_FG,]

        assignment_map = torch.empty_like(fg_mask, dtype=torch.long)
        assignment_map[fg_mask] = L_argmax
        assignment_map[~fg_mask] = -1

        ax.imshow((assignment_map + 1).squeeze().cpu().numpy(), cmap="tab10")
        ax.set_xticks([]), ax.set_yticks([])

        fig.tight_layout()

        # Update prototypes
        prototypes_new = torch.mm(L_optimized.t(), fg_patches_flat_norm)
        prototypes = 0.99 * prototypes + 0.01 * F.normalize(prototypes_new, p=2, dim=-1)
    plt.show()
