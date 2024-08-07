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

from cub_dataset import CUBDataset
from models.backbone import load_backbone
from models.ppnet import get_projection_layer
from models.pp_concept_net import ProtoPConceptNet, ProtoPNetLoss


def train_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, criterion: nn.Module,
                optimizer: optim.Optimizer, summary_writer: SummaryWriter,
                logger: Logger, device: torch.device, epoch_name: str = "warmup"):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    with_concepts = epoch_name == "final"
    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _ = batch
        outputs = model(images, with_concepts)
        class_logits, concept_logits, min_dists, all_dists = outputs
        loss_dict = criterion(outputs=outputs,
                              batch=batch,
                              proto_class_association=model.proto_class_association)

        total_loss = sum(loss_dict.values())
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * dataloader.batch_size

        mca_train(class_logits, labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        summary_writer.add_scalar(f"Loss/{epoch_name}/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {epoch_name} train {k}: {loss_avg:.4f}")
    epoch_acc_train = mca_train.compute().item()
    summary_writer.add_scalar(f"Acc/{epoch_name}/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def val_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, summary_writer: SummaryWriter,
              logger: Logger, device: torch.device, epoch_name: str = "warmup"):
    model.eval()
    mca_val = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)
    with_concepts = epoch_name == "final"
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels, _ = tuple(item.to(device) for item in batch)
            class_logits, concept_logits, min_dists, all_dists = model(images, with_concepts)

            mca_val(class_logits, labels)

    epoch_acc_val = mca_val.compute().item()
    summary_writer.add_scalar(f"Acc/{epoch_name}/val", epoch_acc_val, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} val acc: {epoch_acc_val:.4f}")

    return epoch_acc_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--options", "-o", type=str, nargs="+")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(args.config_path)
    log_dir = Path("logs")
    config = OmegaConf.load(Path(config_path))
    if args.options:
        log_dir /= config_path.stem + "-concept-" + "-".join(args.options)
        config.merge_with_dotlist(args.options)
    else:
        log_dir /= config_path.stem + "-concept"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(config))
    OmegaConf.save(config=config, f=log_dir / "config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    L.seed_everything(42)

    # Load data
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406,),
                    std=(0.229, 0.224, 0.225,))
    ])
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    dataset_train = CUBDataset((dataset_dir / "train_cropped_augmented").as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)

    # Load model
    backbone, backbone_dim = load_backbone(config.model.backbone)
    proj_layers = get_projection_layer(config.model.proj_layers,
                                       first_dim=backbone_dim)

    ppnet = ProtoPConceptNet(backbone, proj_layers, dataset_train.attributes,
                             (2000, 128, 1, 1,), 200)
    criterion = ProtoPNetLoss(l_clst_coef=config.model.l_clst_coef,
                              l_sep_coef=config.model.l_sep_coef)

    print("Projection Layers:")
    print(proj_layers)

    # Load optimizers
    warmup_param_groups = [
        {'params': ppnet.proj.parameters(),
         'lr': config.optim.warmup.proj_lr,
         'weight_decay': config.optim.warmup.proj_weight_decay},
        {'params': ppnet.prototype_vectors,
         'lr': config.optim.warmup.proto_lr},
    ]
    joint_param_groups = [
        {'params': ppnet.backbone.parameters(),
         'lr': config.optim.joint.backbone_lr,
         'weight_decay': config.optim.joint.backbone_weight_decay},
        {'params': ppnet.proj.parameters(),
         'lr': config.optim.joint.proj_lr,
         'weight_decay': config.optim.joint.proj_weight_decay},
        {'params': ppnet.prototype_vectors,
         'lr': config.optim.joint.proto_lr}
    ]

    optimizer_warmup = optim.Adam(warmup_param_groups)
    optimizer_joint = optim.Adam(joint_param_groups)
    lr_scheduler_joint = optim.lr_scheduler.StepLR(optimizer_joint, step_size=5, gamma=0.1)

    final_param_groups = [
        {'params': ppnet.fc_concepts.parameters(),
         'lr': config.optim.final.fc_lr}
    ]
    optimizer_final = optim.Adam(final_param_groups)

    # Prepare for training
    writer = SummaryWriter(log_dir=log_dir.as_posix())
    ppnet.to(device)

    best_epoch, best_val_acc = 0, 0.
    early_stopping_epochs = 8

    epoch_name = "warmup"
    optimizer = optimizer_warmup
    lr_scheduler = None
    for param in ppnet.backbone.parameters():
        param.requires_grad = False
    for param in ppnet.proj.parameters():
        param.requires_grad = True
    ppnet.prototype_vectors.requires_grad = True
    for param in ppnet.fc_concepts.parameters():
        param.requires_grad = False

    # epoch: 0-5 warmup; 5-9 joint; 10-29 final; 30-40 final; 40-60 joint
    for epoch in range(60):
        if epoch in [5, 30]:
            logger.info("Start joint stage...")
            epoch_name = "joint"
            for name, param in ppnet.named_parameters():
                param.requires_grad = "fc_concepts" not in name
            optimizer = optimizer_joint
            lr_scheduler = lr_scheduler_joint
        if epoch in [15, 50]:
            logger.info("Start fine-tuning final layer only...")
            epoch_name = "final"
            for name, param in ppnet.named_parameters():
                param.requires_grad = "fc_concepts" in name
            optimizer = optimizer_final
            lr_scheduler = None

        train_epoch(model=ppnet, dataloader=dataloader_train, epoch=epoch,
                    criterion=criterion, optimizer=optimizer, summary_writer=writer,
                    logger=logger, device=device, epoch_name=epoch_name)

        if lr_scheduler:
            lr_scheduler.step()

        epoch_acc_val = val_epoch(model=ppnet, dataloader=dataloader_test, epoch=epoch,
                                  summary_writer=writer, logger=logger, device=device, epoch_name=epoch_name)

        # Early stopping based on validation accuracy
        if epoch_acc_val > best_val_acc:
            torch.save({k: v.cpu() for k, v in ppnet.state_dict().items()},
                       log_dir / "ppnet_best.pth")
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            logger.info("Best epoch found, model saved!")
        if epoch >= best_epoch + early_stopping_epochs:
            break


if __name__ == '__main__':
    main()
