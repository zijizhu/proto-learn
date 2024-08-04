#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from collections import defaultdict
from logging import Logger
from pathlib import Path

import lightning as L
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from einops import repeat

from cub_dataset import CUBDataset
from models.proto_cls import ProtoNetCLS, Loss


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
        clf_logits = outputs["seg"].sum((-1, -2,))[:, :-1]

        loss_dict = criterion(outputs, labels, model.fc)
        total_loss = sum(v for v in loss_dict.values())

        # total_loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * dataloader.batch_size

        mca_train(clf_logits, labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        # summary_writer.add_scalar(f"Loss/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {epoch_name} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    # summary_writer.add_scalar("Acc/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} train acc: {epoch_acc_train:.4f}")
    
    if epoch in [5, 10, 15]:
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, f"dino_v2_epoch{epoch}.pth")


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
            clf_logits = outputs["class_logits"].sum((-1, -2,))[:, :-1]

            mca_val(clf_logits, labels)

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

    net = ProtoNetCLS()
    criterion = Loss()

    # Load optimizers
    joint_param_groups = [
        {'params': net.proj.parameters(),
         'lr': 0.01,
         'weight_decay': 0.0001},
    ]
    for params in net.backbone.parameters():
        params.requires_grad = False
    
    optimizer = optim.Adam(joint_param_groups)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Prepare for training
    # writer = SummaryWriter(log_dir=log_dir.as_posix())
    net.to(device)
    criterion.to(device)

    best_epoch, best_val_acc = 0, 0.

    # epoch: 0-9 joint; 10-29 final; 30-40 joint; 40-60 final
    for epoch in range(30):
        train_epoch(model=net, dataloader=dataloader_train, epoch=epoch,
                    criterion=criterion, optimizer=optimizer,
                    logger=logger, device=device)

        if lr_scheduler:
            lr_scheduler.step()

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