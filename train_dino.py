#!/usr/bin/env python3
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "dinov2"))
import logging
from collections import defaultdict
from logging import Logger
from pathlib import Path

import lightning as L
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from models.dino import ProtoDINO, ProtoPNetLoss
from models.backbone import DINOv2BackboneExpanded, load_adapter
from cub_dataset import CUBDataset
from utils.visualization import visualize_prototype_assignments, visualize_topk_prototypes
from utils.config import setup_config_and_logging
from models.utils import print_parameters


def train_epoch(optimize_prototypes: bool, model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
                optimizer: optim.Optimizer | None, writer: SummaryWriter,
                logger: Logger, device: torch.device, debug: bool = False):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _, sample_indices = batch
        outputs = model(images, labels=labels, optimize_prototypes=optimize_prototypes)

        if criterion is not None and optimizer is not None:
            loss_dict = criterion(outputs, batch)
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader.batch_size

        mca_train(outputs["class_logits"], labels)

        if debug and i == 0:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, writer, epoch, epoch_name="train")
            visualize_prototype_assignments(outputs, labels, writer, epoch, epoch_name="train", figsize=(8, 10,))

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        writer.add_scalar(f"Loss/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} train {k}: {loss_avg:.4f}")

    epoch_acc_train = mca_train.compute().item()
    writer.add_scalar("Acc/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def val_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, writer: SummaryWriter,
              logger: Logger, device: torch.device, debug: bool = False):
    model.eval()
    mca_val = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _, sample_indices = batch
        outputs = model(images, labels=labels, optimize_prototypes=False)

        if debug and i == 0:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, writer, epoch, epoch_name="val")
            visualize_prototype_assignments(outputs, labels, writer, epoch, epoch_name="val", figsize=(10, 10,))

        mca_val(outputs["class_logits"], labels)

    epoch_acc_val = mca_val.compute().item()
    writer.add_scalar("Acc/val", epoch_acc_val, epoch)
    logger.info(f"EPOCH {epoch} val acc: {epoch_acc_val:.4f}")

    return epoch_acc_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir = setup_config_and_logging(name="train", base_log_dir="logs_new")

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
    training_set_path = "train_cropped_augmented" if cfg.dataset.augmentation else "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)

    backbone = DINOv2BackboneExpanded(name=cfg.model.name, n_splits=cfg.model.n_splits)
    net = ProtoDINO(
        backbone=backbone,
        dim=backbone.dim,
        pooling_method=cfg.model.pooling_method,
        cls_head=cfg.model.cls_head,
        pca_compare=cfg.model.pca_compare
    )

    param_groups = []
    if cfg.model.tuning == "block_expansion":
        backbone_params = list(p for name, p in net.backbone.named_parameters() if name in net.backbone.learnable_param_names)
        param_groups += [{'params': backbone_params, 'lr': cfg.optim.backbone_lr}]
    param_groups += [{'params': net.sa, 'lr': cfg.optim.fc_lr}] if cfg.model.cls_head == "sa" else []
    
    criterion = ProtoPNetLoss(**cfg.model.losses)
    optimizer = optim.SGD(param_groups, momentum=0.9)

    net.to(device)
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    best_epoch, best_val_acc = 0, 0.
    for epoch in range(cfg.optim.epochs):
        is_fine_tuning = epoch in cfg.optim.fine_tuning_epochs
        optimizing_prototypes = not is_fine_tuning

        if is_fine_tuning:
            logger.info("Start fine-tuning...")
            for name, param in net.named_parameters():
                param.requires_grad = "backbone" not in name

            for name, param in net.backbone.dino.named_parameters():
                param.requires_grad = name in net.backbone.learnable_param_names
            else:
                raise AttributeError

        else:
            for params in net.parameters():
                params.requires_grad = False

        print_parameters(net=net, logger=logger)
        debug = epoch in cfg.debug.epochs

        train_epoch(
            optimize_prototypes=optimizing_prototypes,
            model=net,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            writer=writer,
            logger=logger,
            device=device,
            debug=debug
        )

        epoch_acc_val = val_epoch(model=net, dataloader=dataloader_test, epoch=epoch,
                                  writer=writer, logger=logger, device=device,  debug=debug)

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, log_dir / "dino_v2_proto.pth")
            logger.info("Best epoch found, model saved!")

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
