#!/usr/bin/env python3
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

from models.backbone import DINOv2BackboneExpanded, MaskCLIP, DINOv2Backbone
from cub_dataset import CUBDataset, CUBFewShotDataset
from utils.visualization import visualize_prototype_assignments, visualize_topk_prototypes
from utils.config import setup_config_and_logging
from models.utils import print_parameters


def train_epoch(model: nn.Module, criterion: nn.Module | None, dataloader: DataLoader, epoch: int,
                optimizer: optim.Optimizer | None, writer: SummaryWriter,
                logger: Logger, device: torch.device, debug: bool = False):
    model.train()
    running_losses = defaultdict(float)
    mca_train = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels, sample_indices = batch

        outputs = model(images, labels=labels)

        if criterion is not None and optimizer is not None:
            loss_dict = criterion(outputs, batch)
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader.batch_size

        mca_train(outputs["class_logits"], labels)

        # if debug and i == 0:
        #     batch_size, _, input_size, input_size = images.shape
        #     batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
        #     visualize_topk_prototypes(outputs, batch_im_paths, writer, step=epoch, input_size=input_size,
        #                               tag_fmt_str="Training first batch top{topk} prototypes/epoch {step}/{idx}")
        #     visualize_prototype_assignments(outputs, writer, step=epoch,
        #                                     tag=f"Training first batch prototype assignments/epoch {epoch}")

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
        images, labels, sample_indices = batch

        outputs = model(images, labels=labels)

        mca_val(outputs["class_logits"], labels)

        # if debug and i == 0:
        #     batch_size, input_size, input_size = images.shape
        #     batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
        #     visualize_topk_prototypes(outputs, batch_im_paths, writer, step=epoch, input_size=input_size,
        #                               tag_fmt_str="Training epoch {step} batch 0 top{topk} prototypes/{idx}")
        #     visualize_prototype_assignments(outputs, labels, writer, step=epoch,
        #                                     tag=f"Validation epoch {epoch} batch {i} prototype assignments")

    epoch_acc_val = mca_val.compute().item()
    writer.add_scalar("Acc/val", epoch_acc_val, epoch)
    logger.info(f"EPOCH {epoch} val acc: {epoch_acc_val:.4f}")

    return epoch_acc_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir, resume_ckpt = setup_config_and_logging(name="train", base_log_dir="logs")

    logger = logging.getLogger(__name__)

    L.seed_everything(cfg.seed)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    n_classes = 200
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    training_set_path = "train_cropped_augmented" if cfg.dataset.augmentation else "train_cropped"
    dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                               transforms=transforms)
    dataset_train_few_shot  = CUBFewShotDataset((dataset_dir / training_set_path).as_posix(),
                                 n_samples_per_class=10,
                                 attribute_label_path=attribute_labels_path,
                                 transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              transforms=transforms)
    if cfg.get("few_shot", False):
        dataloader_train = DataLoader(dataset=dataset_train_few_shot, batch_size=128, num_workers=8, shuffle=True)
    else:
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)

    if "dino" in cfg.model.name:
        if cfg.model.n_splits and cfg.model.n_splits > 0:
            backbone = DINOv2BackboneExpanded(name=cfg.model.name, n_splits=cfg.model.n_splits)
        else:
            backbone = DINOv2Backbone(name=cfg.model.name)
        dim = backbone.dim
    elif cfg.model.name.lower().startswith("clip"):
        backbone = MaskCLIP(name=cfg.model.name.split("-", 1)[1])
        dim = 512
    else:
        raise NotImplementedError("Backbone must be one of dinov2 or clip.")
    
    if cfg.model.adapter:
        from models.dino_adapt import ProtoDINO, ProtoPNetLoss, PaPr, PCA
    else:
        from models.dino import ProtoDINO, ProtoPNetLoss, PaPr, PCA
    
    assert cfg.model.fg_extractor in ["PCA", "PaPr"]
    if cfg.model.fg_extractor == "PaPr":
        fg_extractor = PaPr(bg_class=n_classes, **cfg.model.fg_extractor_args)
    else:
        fg_extractor = PCA(bg_class=n_classes, **cfg.model.fg_extractor_args)

    net = ProtoDINO(
        backbone=backbone,
        dim=dim,
        # fg_extractor=fg_extractor
        fg_extractor=None,
        n_prototypes=cfg.model.n_prototypes,
        gamma=cfg.model.get("gamma", 0.99),
        temperature=cfg.model.temperature,
        pooling_method=cfg.model.pooling_method,
        cls_head=cfg.model.cls_head,
        sa_init=cfg.model.sa_init
    )
    if resume_ckpt:
        state_dict = torch.load(resume_ckpt, map_location="cpu")
        net.load_state_dict(state_dict)

    criterion = ProtoPNetLoss(**cfg.model.losses, n_prototypes=cfg.model.n_prototypes)

    net.to(device)
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    best_epoch, best_val_acc = 0, 0.
    for epoch in range(cfg.optim.epochs):
        is_fine_tuning = epoch in cfg.optim.fine_tuning_epochs
        is_debugging = epoch in cfg.debug.epochs

        if is_fine_tuning:
            logger.info("Start fine-tuning...")
            for name, param in net.named_parameters():
                param.requires_grad = ("backbone" not in name) and ("fg_extractor" not in name)
            
            if cfg.model.tuning is not None:
                net.backbone.set_requires_grad()
            param_groups = []
            # param_groups = [{'params': net.backbone.learnable_parameters(), 'lr': cfg.optim.backbone_lr}] if (cfg.model.n_splits != 0 and cfg.model.tuning is not None) else []
            param_groups += [{'params': net.learnable_prototypes, 'lr': cfg.optim.adapter_lr}] if cfg.model.adapter else []  # DEBUG
            param_groups += [{'params': net.adapters.parameters(), 'lr': cfg.optim.adapter_lr}] if cfg.model.adapter else []
            param_groups += [{'params': net.sa, 'lr': cfg.optim.sa_lr}] if cfg.model.cls_head == "sa" else []
            optimizer = optim.SGD(param_groups, momentum=0.9)
            if cfg.model.get("always_optimize_prototypes", False):
                net.optimizing_prototypes = True
            else:
                net.optimizing_prototypes = False
        else:
            for params in net.parameters():
                params.requires_grad = False
            optimizer = None
            net.optimizing_prototypes = True

        if (epoch > 0) or (resume_ckpt is not None):
            net.initializing = False

        print_parameters(net=net, logger=logger)
        logger.info(f"net.initializing: {net.initializing}")
        logger.info(f"net.optimizing_prototypes: {net.optimizing_prototypes}")

        train_epoch(
            model=net,
            criterion=criterion if is_fine_tuning else None,
            dataloader=dataloader_train,
            epoch=epoch,
            optimizer=optimizer if is_fine_tuning else None,
            writer=writer,
            logger=logger,
            device=device,
            debug=is_debugging
        )

        epoch_acc_val = val_epoch(model=net, dataloader=dataloader_test, epoch=epoch,
                                  writer=writer, logger=logger, device=device)

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, log_dir / "dino_v2_proto.pth")
            logger.info("Best epoch found, model saved!")

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
