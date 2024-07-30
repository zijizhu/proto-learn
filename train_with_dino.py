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
from projection import project_prototypes
from models.ppnet_dino import ProtoPNetDINO, get_projection_layer, ProtoPNetLoss


def train_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, criterion: nn.Module,
                optimizer: optim.Optimizer, summary_writer: SummaryWriter,
                logger: Logger, device: torch.device, epoch_name: str = "joint"):

    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)
        images, labels, _ = batch
        outputs = model(images)
        logits = outputs["logits"]
        loss_dict = criterion(outputs=outputs,
                              batch=batch,
                              proto_class_association=model.proto_class_association,
                              fc_weights=model.fc.weight)

        total_loss = sum(v for k, v in loss_dict.items() if not k.startswith("_"))
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for k, v in loss_dict.items():
            running_losses[k] += v.item() * dataloader.batch_size

        mca_train(logits, labels)

    for k, v in running_losses.items():
        loss_avg = v / len(dataloader.dataset)
        summary_writer.add_scalar(f"Loss/{k}", loss_avg, epoch)
        logger.info(f"EPOCH {epoch} {epoch_name} train {k}: {loss_avg:.4f}")

    sep_clst_sum = (running_losses["l_clst"] + running_losses["l_sep"]) / len(dataloader.dataset)
    summary_writer.add_scalar("Loss/sep_clst_sum", sep_clst_sum, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} train sep_clst_sum: {sep_clst_sum:.4f}")

    epoch_acc_train = mca_train.compute().item()
    summary_writer.add_scalar("Acc/train", epoch_acc_train, epoch)
    logger.info(f"EPOCH {epoch} {epoch_name} train acc: {epoch_acc_train:.4f}")


@torch.inference_mode()
def val_epoch(model: nn.Module, dataloader: DataLoader, epoch: int, summary_writer: SummaryWriter,
              logger: Logger, device: torch.device, epoch_name: str = "joint"):
    model.eval()
    mca_val = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, labels, _ = tuple(item.to(device) for item in batch)
            outputs = model(images)
            logits = outputs["logits"]

            mca_val(logits, labels)

    epoch_acc_val = mca_val.compute().item()
    summary_writer.add_scalar("Acc/val", epoch_acc_val, epoch)
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
        log_dir /= config_path.stem + "-" + "-".join(args.options)
        config.merge_with_dotlist(args.options)
    else:
        log_dir /= config_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Configuration:")
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

    L.seed_everything(config.optim.seed)

    # Construct augmentations
    input_size = config.model.input_size
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    test_transforms = T.Compose([
            T.Resize((input_size, input_size,)),
            T.ToTensor(),
            normalize
        ])
    if config.model.augmentation:
        attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
        train_transforms = test_transforms
    else:
        attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
        train_transforms = T.Compose([
            T.Resize((input_size, input_size,)),
            T.RandomAffine(degrees=(-25, 25), shear=15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    
    # Load datasets and dataloaders
    dataset_dir = Path("datasets") / "cub200_cropped"
    training_set_path = "train_cropped_augmented" if config.model.augmentation else "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=train_transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=test_transforms)
    dataset_projection = CUBDataset((dataset_dir / "train_cropped").as_posix(),
                                    attribute_labels_path.as_posix(),
                                    transforms=test_transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)
    dataloader_projection = DataLoader(dataset=dataset_projection, batch_size=75, num_workers=8, shuffle=False)

    # Load model
    proj_layers = get_projection_layer(config.model.proj_layers)
    
    ppnet = ProtoPNetDINO("dinov2_vitb14_reg",
                          proj_layers=proj_layers,
                          prototype_shape=tuple(config.model.prototype_shape),
                          num_classes=200)
    criterion = ProtoPNetLoss(l_clst_coef=config.model.l_clst_coef,
                              l_sep_coef=config.model.l_sep_coef,
                              l_l1_coef=config.model.l_l1_coef)

    print("Projection Layers:")
    print(proj_layers)

    # Load optimizers
    joint_param_groups = [
        {'params': ppnet.proj.parameters(),
         'lr': config.optim.joint.proj_lr,
         'weight_decay': config.optim.joint.proj_weight_decay},
        {'params': ppnet.prototype_vectors,
         'lr': config.optim.joint.proto_lr}
    ]
    
    optimizer_joint = optim.Adam(joint_param_groups)
    lr_scheduler_joint = optim.lr_scheduler.StepLR(optimizer_joint, step_size=5, gamma=0.1)

    if config.optim.final:
        final_param_groups = [
            {'params': ppnet.fc.parameters(),
            'lr': config.optim.final.fc_lr}
        ]
        optimizer_final = optim.Adam(final_param_groups)
    
    # Prepare for training
    writer = SummaryWriter(log_dir=log_dir.as_posix())
    ppnet.to(device)

    best_epoch, best_val_acc = 0, 0.

    # epoch: 0-9 joint; 10-29 final; 30-40 joint; 40-60 final
    for epoch in range(60):
        if epoch in config.optim.joint.epoch_start:
            epoch_name = "joint"
            for param in ppnet.backbone.parameters():
                param.requires_grad = False
            for param in ppnet.proj.parameters():
                param.requires_grad = True
            ppnet.prototype_vectors.requires_grad = True
            for param in ppnet.fc.parameters():
                param.requires_grad = False
            optimizer = optimizer_joint
            lr_scheduler = lr_scheduler_joint

        elif config.optim.final and epoch in config.optim.final.epoch_start:
            logger.info(f"Perform prototype projection before epoch {epoch}...")
            projection_results = project_prototypes(ppnet, dataloader_projection, device=device)
            torch.save(projection_results, log_dir / f"projection_results_epoch{epoch}.pth")
            writer.add_histogram("Projection_Min_Dists", projection_results["min_l2_dists"], epoch)

            epoch_name = "final"
            for name, param in ppnet.named_parameters():
                param.requires_grad = "fc" in name
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

    torch.save({k: v.cpu() for k, v in ppnet.state_dict().items()},
                       log_dir / "ppnet_final.pth")
    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
