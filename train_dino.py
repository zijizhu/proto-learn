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
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from models.dino import ProtoDINO, ProtoPNetLoss
from models.backbone import DINOv2BackboneExpanded, load_projection
from cub_dataset import CUBDataset
from utils.visualization import visualize_prototype_assignments, visualize_topk_prototypes
from utils.config import setup_config_and_logging
from models.utils import get_cosine_schedule_with_warmup


def train_epoch(model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
                optimizer: optim.Optimizer | None, writer: SummaryWriter,
                logger: Logger, device: torch.device, debug: bool = False):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _, sample_indices = batch
        outputs = model(images, labels=labels, debug=True, use_gumbel=False)

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
        outputs = model(images, labels=labels, debug=True, use_gumbel=False)

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
    
    config, log_dir = setup_config_and_logging(base_log_dir="logs_new")

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
    training_set_path = "train_cropped_augmented" if config["dataset"]["augmentation"] else "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)

    n_splits = 1
    neck = load_projection(config["model"])
    backbone = DINOv2BackboneExpanded(name=config["model"]["name"], n_splits=config["model"]["n_splits"])
    net = ProtoDINO(backbone=backbone,
                    neck=neck,
                    pooling_method=config["model"]["pooling_method"],
                    cls_head=config["model"]["cls_head"], dim=backbone.dim)

    for params in net.parameters():
        params.requires_grad = False
    
    train_fc = (config["model"]["cls_head"] in ["fc", "sa"]) or n_splits > 0
    optimizer = None
    scheduler = None
    # criterion = ProtoPNetLoss(l_clst_coef=-0.08, l_sep_coef=0.008, l_l1_coef=0) if train_fc else None
    criterion = ProtoPNetLoss(l_clst_coef=0, l_sep_coef=0, l_l1_coef=0) if train_fc else None
    
    net.backbone._check()
    
    net.to(device)
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    best_epoch, best_val_acc = 0, 0.
    fc_start_epoch = config["optim"].get("fc_start_epoch", None)
    for epoch in range(config["optim"]["epochs"]):
        epoch_train_fc = train_fc and (epoch >= fc_start_epoch)
        if train_fc and (epoch == fc_start_epoch):
            print("start fine-tuning")
            net.freeze_prototypes = True
            if config["model"]["cls_head"] == "sa":
                net.sa.requires_grad = True
            for param in net.neck.parameters():
                param.requires_grad = True
            
            param_groups = [{'params': net.neck.parameters(), 'lr': config["optim"]["backbone_lr"]}]
            param_groups += [{'params': net.sa, 'lr': config["optim"]["fc_lr"]}] if config["model"]["cls_head"] == "sa" else []
            optimizer = optim.SGD(param_groups, momentum=0.9)
            if config["optim"]["scheduler"] == "cosine":
                scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=int(len(dataset_train) / 128) * config["optim"]["fc_start_epoch"],
                                                            num_training_steps=47 * config["optim"]["epochs"])
            else:
                scheduler = None

        debug = epoch in config["debug"]["epochs"]

        train_epoch(
            model=net,
            criterion=criterion if epoch_train_fc else None,
            dataloader=dataloader_train, epoch=epoch,
            optimizer=optimizer if epoch_train_fc else None,
            writer=writer,
            logger=logger,
            device=device,
            debug=debug)

        if scheduler is not None:
            scheduler.step()

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
