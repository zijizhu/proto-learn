#!/usr/bin/env python3
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
from torch import nn, optim
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
    for b, (prototype_logits, saliency_maps, im_path) in enumerate(tqdm(zip(batch_prototype_logits, batch_saliency_maps, batch_im_paths), total=batch_size)):

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
        writer.add_image(f"{class_name}/{fname} top {topk} prototypes", F.pil_to_tensor(fig_image))

    return figures


def train_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, criterion: nn.Module,
                optimizer: optim.Optimizer,
                logger: Logger, device: torch.device, epoch_name: str = "joint"):

    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)


    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _ = batch
        outputs = model(images, labels=labels)

        # loss_dict = criterion(outputs, labels, model.associations)
        # total_loss = sum(v for v in loss_dict.values())

        # total_loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        # total_loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        # for k, v in loss_dict.items():
        #     running_losses[k] += v.item() * dataloader.batch_size
        mca_train(outputs["class_logits"][:, :-1], labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        # summary_writer.add_scalar(f"Loss/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {epoch_name} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    # summary_writer.add_scalar("Acc/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} train acc: {epoch_acc_train:.4f}")
    
    # if epoch in [5, 10, 15]:
    #     torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"dino_v2_epoch{epoch}.pth")


@torch.inference_mode()
def val_epoch(model: nn.Module, dataloader: DataLoader, epoch: int,
              logger: Logger, device: torch.device, epoch_name: str = "joint"):
    model.eval()
    mca_val = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = tuple(item.to(device) for item in batch)
            images, labels, _ = batch
            outputs = model(images)
            # clf_logits = outputs["class_logits"].sum((-1, -2,))[:, :-1]

            mca_val(outputs["class_logits"][:, :-1], labels)

    epoch_acc_val = mca_val.compute().item()
    logger.info(f"EPOCH {epoch} {epoch_name} val acc: {epoch_acc_val:.4f}")
    
    return epoch_acc_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path("tmp.log")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir.as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    L.seed_everything(42)

    # Construct augmentations
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])
    
    # Load datasets and dataloaders
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    training_set_path = "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)
    # dataset_projection = CUBDataset((dataset_dir / "train_cropped").as_posix(),
    #                                 attribute_labels_path.as_posix(),
    #                                 transforms=transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)
    # dataloader_projection = DataLoader(dataset=dataset_projection, batch_size=75, num_workers=8, shuffle=False)

    net = ProtoDINO()

    # Load optimizers
    # joint_param_groups = [
    #     {'params': net.proj.parameters(),
    #      'lr': 0.01,
    #      'weight_decay': 0.0001},
    # ]
    for params in net.backbone.parameters():
        params.requires_grad = False
    
    # optimizer = optim.Adam(joint_param_groups)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Prepare for training
    # writer = SummaryWriter(log_dir=log_dir.as_posix())
    net.to(device)
    # criterion.to(device)

    best_epoch, best_val_acc = 0, 0.

    # epoch: 0-9 joint; 10-29 final; 30-40 joint; 40-60 final
    for epoch in range(30):
        train_epoch(model=net, dataloader=dataloader_train, epoch=epoch,
                    criterion=None, optimizer=None,
                    logger=logger, device=device)

        # if lr_scheduler:
        #     lr_scheduler.step()

        epoch_acc_val = val_epoch(model=net, dataloader=dataloader_test, epoch=epoch,
                                  logger=logger, device=device)

        # Early stopping based on validation accuracy
        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, "dino_v2_proto.pth")
            logger.info("Best epoch found, model saved!")

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
