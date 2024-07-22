import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from cub_dataset import CUBDataset
from models.backbone import load_backbone
from models.ppnet import ProtoPNet, get_projection_layer


def sample_images_from_classes(dataset: ImageFolder, num_samples_per_class: int = 5):
    sampled_im_paths = []
    for class_name in dataset.classes:
        class_im_paths = [im_path for im_path, _ in dataset.samples if class_name in im_path]
        sampled_class_im_paths = random.sample(class_im_paths, num_samples_per_class)
        sampled_im_paths.extend(sampled_class_im_paths)

    return sampled_im_paths


def distance_to_similarity(distances: torch.Tensor,
                           activation_fn: str = "log",
                           eps: float = 1e-4) -> torch.Tensor:
    if activation_fn == 'log':
        return torch.log((distances + 1) / (distances + eps))
    else:
        return -distances


def overlay_attn_map(attn_map: torch.Tensor, im: Image.Image | torch.Tensor | np.ndarray):
    if isinstance(im, torch.Tensor):
        assert im.dtype == torch.uint8
        im = F.to_pil_image(im)
    elif isinstance(im, np.ndarray):
        assert im.dtype == np.uint8
        Image.fromarray(im).convert("RGB")
    im = np.array(im.resize((224, 224), Image.BILINEAR))

    attn_map_np = attn_map.numpy()
    max_val, min_val = np.max(attn_map_np), np.min(attn_map_np)
    attn_map_np = (attn_map_np - min_val) / (max_val - min_val)
    heatmap = cv2.applyColorMap((255 * attn_map_np).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.5 * heatmap + 0.5 * im).astype(np.uint8)


def get_high_activation_bbox(attn_map: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, ...]:
    min_val, max_val = attn_map.min(), attn_map.max()
    attn_map = (attn_map - min_val) / (max_val - min_val)
    high_activation_mask = attn_map >= threshold

    high_activation_coords = high_activation_mask.nonzero()
    y_min, y_max = torch.min(high_activation_coords[:, 0]), torch.max(high_activation_coords[:, 0])
    x_min, x_max = torch.min(high_activation_coords[:, 1]), torch.max(high_activation_coords[:, 1])

    assert x_min <= x_max and y_min <= y_max
    return x_min, y_min, x_max, y_max


@torch.inference_mode()
def visualize_top_prototypes(im_path: str,
                             transforms: T.Compose,
                             net: nn.Module,
                             dataset: ImageFolder,
                             train_proto_nearest_patches: torch.Tensor,
                             train_proto_dists: torch.Tensor,
                             device: torch.device,
                             with_concepts: bool = False):
    im_raw = Image.open(im_path).convert("RGB")
    im_transformed = transforms(im_raw).unsqueeze(0).to(device)
    if with_concepts:
        outputs = net(im_transformed, with_concepts)
        outputs = tuple(item.detach().cpu().squeeze() for item in outputs)
        logits, concept_logits, min_dists, attn_maps = outputs
    else:
        outputs = net.inference(im_transformed)
        outputs = tuple(item.detach().cpu().squeeze() for item in outputs)
        logits, min_dists, attn_maps = outputs

    topk_negative_dists, topk_proto_indices = torch.topk(-min_dists, k=5)
    least_k_min_dists = -topk_negative_dists
    topk_similarities = distance_to_similarity(least_k_min_dists)

    # Process attention maps
    attn_maps = attn_maps.squeeze()[topk_proto_indices]
    attn_maps = distance_to_similarity(attn_maps)
    attn_maps = F.resize(attn_maps, [224, 224], F.InterpolationMode.BICUBIC)

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(9, 12))
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])

    for i, (amap, proto_index, ax_row, dist, similarity) in enumerate(
            zip(attn_maps, topk_proto_indices, axes, least_k_min_dists, topk_similarities)):
        # Read image of nearest training sample
        nearest_train_sample_index = train_proto_nearest_patches[proto_index, 1, 0].item()
        nearest_train_sample_im_path, _ = dataset.samples[nearest_train_sample_index]
        nearest_train_sample_im_pt = read_image(nearest_train_sample_im_path, ImageReadMode.RGB)
        nearest_train_sample_im_pt = F.resize(nearest_train_sample_im_pt,
                                              [224, 224],
                                              F.InterpolationMode.BICUBIC)

        # Process attention map and overlay it on image
        nearest_train_sample_attn_map = train_proto_dists[nearest_train_sample_index, proto_index]
        nearest_train_sample_attn_map = distance_to_similarity(nearest_train_sample_attn_map)
        nearest_train_sample_attn_map = F.resize(nearest_train_sample_attn_map.unsqueeze(0), [224, 224],
                                                 F.InterpolationMode.BICUBIC).squeeze()
        nearest_train_sample_overlay = overlay_attn_map(nearest_train_sample_attn_map, nearest_train_sample_im_pt)

        # Get bounding box of prototypical part and draw on image
        bbox = get_high_activation_bbox(nearest_train_sample_attn_map)
        nearest_train_sample_bbox = draw_bounding_boxes(nearest_train_sample_im_pt, torch.tensor([bbox]),
                                                        colors="red", width=2)

        ax_row[0].imshow(overlay_attn_map(amap, im_raw))
        ax_row[0].title.set_text(f"Top {i + 1} Distance: {dist.item():.4f}, Similarity: {similarity.item():.4f}")
        ax_row[0].title.set_fontsize(10)

        ax_row[1].imshow(F.to_pil_image(nearest_train_sample_bbox))
        ax_row[1].title.set_text(f"Class: {Path(nearest_train_sample_im_path).parts[-2]}")
        ax_row[1].title.set_fontsize(10)

        ax_row[2].imshow(nearest_train_sample_overlay)
        ax_row[2].title.set_text(f"{proto_index % 10}th prototype of class {proto_index // 10}")
        ax_row[2].title.set_fontsize(10)

        fig.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path)
    parser.add_argument("--num_samples_per_class", type=int, default=5)

    args = parser.parse_args()

    with_concepts = "concept" in args.log_dir.as_posix()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.log_dir / "config.yaml")
    print(OmegaConf.to_yaml(config))

    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406,),
                    std=(0.229, 0.224, 0.225,))
    ])

    # Load datasets
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    dataset_train = CUBDataset((dataset_dir / "train_cropped").as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)

    # Load model
    backbone, dim = load_backbone(backbone_name=config.model.backbone)
    proj_layers = get_projection_layer(config.model.proj_layers, first_dim=dim)
    ppnet = ProtoPNet(backbone, proj_layers, (2000, 128, 1, 1,), 200)
    state_dict = torch.load(args.log_dir / "ppnet_best.pth")
    ppnet.load_state_dict(state_dict)

    train_proto_nearest_patches = torch.load(args.log_dir / "train_proto_nearest_patches.pth")
    train_proto_dists = torch.load(args.log_dir / "train_proto_dists.pth")

    ppnet.eval()
    ppnet.to(device=device)

    sampled_im_paths = sample_images_from_classes(dataset_test, num_samples_per_class=args.num_samples_per_class)

    writer = SummaryWriter(log_dir=args.log_dir.as_posix())

    for im_path in tqdm(sampled_im_paths):
        fig = visualize_top_prototypes(im_path,
                                       transforms,
                                       ppnet,
                                       dataset_train,
                                       train_proto_nearest_patches,
                                       train_proto_dists,
                                       device,
                                       with_concepts)
        writer.add_figure("/".join(Path(im_path).parts[-2:]), fig)


if __name__ == "__main__":
    main()
