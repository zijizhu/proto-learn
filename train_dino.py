#!/usr/bin/env python3
import argparse
import logging
import sys
from collections import defaultdict
from logging import Logger
from pathlib import Path

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from cub_dataset import CUBDataset
from models.dino import ProtoDINO


def overlay_attn_map(attn_map: np.ndarray, im: Image.Image):
    """
    attn_map: np.ndarray of shape (H, W,), same as im, values can be unormalized
    im: a PIL image of width H and height W
    """
    im = np.array(im.resize((224, 224), Image.BILINEAR))

    max_val, min_val = np.max(attn_map), np.min(attn_map)
    attn_map = (attn_map - min_val) / (max_val - min_val)
    heatmap = cv2.applyColorMap((255 * attn_map).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return (0.5 * heatmap + 0.5 * im).astype(np.uint8)


def visualize_topk_prototypes(batch_outputs: dict[str, torch.Tensor],
                              batch_im_paths: list[str],
                              writer: SummaryWriter,
                              epoch: int,
                              *,
                              topk: int = 5,
                              input_size: int = 224,
                              latent_size: int = 16):
    batch_size, C, K = batch_outputs["image_prototype_logits"].shape
    b = 0
    H = W = latent_size

    batch_prototype_logits = rearrange(batch_outputs["image_prototype_logits"], "B C K -> B (C K)")
    batch_saliency_maps = rearrange(batch_outputs["patch_prototype_logits"], "B (H W) C K -> B H W C K", H=H, W=W)
    
    figures = []
    for b, (prototype_logits, saliency_maps, im_path) in enumerate(zip(batch_prototype_logits, batch_saliency_maps, batch_im_paths)):

        logits, indices = prototype_logits.topk(k=topk, dim=-1)
        indices_C, indices_K = torch.unravel_index(indices=indices, shape=(C, K,))
        topk_maps = saliency_maps[:, :, indices_C, indices_K]
        
        overlayed_images = []
        src_im = Image.open(im_path).convert("RGB").resize((input_size, input_size,))

        topk_maps_resized_np = cv2.resize(topk_maps.cpu().numpy(),
                                          (input_size, input_size,),
                                          interpolation=cv2.INTER_LINEAR)
        
        overlayed_images = [overlay_attn_map(topk_maps_resized_np[:, :, i], src_im) for i in range(topk)]  # shaspe: [topk, input_size, input_size, 3]
        
        fig, axes = plt.subplots(1, topk, figsize=(topk+2, 2))
        for ax, im, c, k in zip(axes.flat, overlayed_images, indices_C.tolist(), indices_K.tolist()):
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{c}/{k}")

        class_name, fname = Path(im_path).parts[-2:]
        fig.suptitle(f"{class_name}/{fname} top {topk} prototypes")
        fig.tight_layout()
        
        fig.canvas.draw()
        fig_image = Image.frombuffer('RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig=fig)
        
        figures.append(fig_image)
        writer.add_image(f"Epoch {epoch} top {topk} prototypes/{b}", F.pil_to_tensor(fig_image))

    return figures


def visualize_prototype_assignments(outputs: dict[str, torch.Tensor], labels: torch.Tensor, writer: SummaryWriter,
                                    epoch: int, figsize: tuple[int, int] = (8, 10,)):
    patch_labels = outputs["pseudo_patch_labels"].clone()  # shape: [B, H, W,]
    L_c_dict = {c: L_c.detach().clone() for c, L_c in outputs["L_c_assignment"].items()}

    nrows, ncols = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10,))

    for b, (c, ax) in enumerate(zip(labels.cpu().tolist(), axes.flat)):
        patch_labels_b = patch_labels[b, :, :]  # shape: [H, W,], dtype: torch.long
        fg_mask_b = patch_labels_b != 200  # shape: [H, W,], dtype: bool

        num_foreground_pixels = fg_mask_b.sum().cpu().item()

        L_c_i = L_c_dict[c][:num_foreground_pixels]
        L_c_dict[c] = L_c_dict[c][num_foreground_pixels:]
        L_c_i_argmax = L_c_i.argmax(-1)  # shape: [N,]

        assignment_map = torch.empty_like(fg_mask_b, dtype=torch.long)
        assignment_map[fg_mask_b] = L_c_i_argmax
        assignment_map[~fg_mask_b] = -1

        ax.imshow((assignment_map + 1).squeeze().cpu().numpy(), cmap="tab10")
        ax.set_xticks([]), ax.set_yticks([])
    
    fig.tight_layout()
    fig.canvas.draw()
    fig_image = Image.frombuffer('RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig=fig)
    
    writer.add_image("Batch prototype assignment", F.pil_to_tensor(fig_image), global_step=epoch)
    
    return fig_image


def train_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, writer: SummaryWriter,
                logger: Logger, device: torch.device, debug: bool = False):

    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)


    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _, sample_indices = batch
        outputs = model(images, labels=labels, debug=True, use_gumbel=False)

        mca_train(outputs["class_logits"], labels)
        
        if debug and i == 0:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, writer, epoch)
            visualize_prototype_assignments(outputs, labels, writer, epoch)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        # summary_writer.add_scalar(f"Loss/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    # summary_writer.add_scalar("Acc/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def val_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, writer: SummaryWriter,
              logger: Logger, device: torch.device, debug: bool = False):
    model.eval()
    mca_val = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _, sample_indices = batch
        outputs = model(images)

        if debug and i == 0:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, writer, epoch)
            visualize_prototype_assignments(outputs, labels, writer, epoch, figsize=(10, 10,))

        mca_val(outputs["class_logits"], labels)

    epoch_acc_val = mca_val.compute().item()
    logger.info(f"EPOCH {epoch} val acc: {epoch_acc_val:.4f}")
    
    return epoch_acc_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path("logs") / args.log_dir
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    L.seed_everything(42)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    training_set_path = "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)

    net = ProtoDINO()
    for params in net.backbone.parameters():
        params.requires_grad = False
    net.to(device)
    
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    best_epoch, best_val_acc = 0, 0.
    for epoch in range(11):
        debug = epoch in [0, 5, 10]
        train_epoch(model=net, dataloader=dataloader_train, epoch=epoch,
                    writer=writer, logger=logger, debug=debug, device=device)

        epoch_acc_val = val_epoch(model=net, dataloader=dataloader_test, epoch=epoch,
                                  writer=writer, logger=logger, device=device)

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, "dino_v2_proto.pth")
            logger.info("Best epoch found, model saved!")

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
