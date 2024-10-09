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

from models.dino import ProtoDINO, ProtoPNetLoss, PaPr, PCA
from models.backbone import DINOv2BackboneExpanded, MaskCLIP, DINOv2Backbone
from cub_dataset import CUBDataset, CUBFewShotDataset, CUBConceptDataset
from utils.visualization import visualize_prototype_assignments, visualize_topk_prototypes, visualize_gt_class_prototypes
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
        images, labels  = batch[:2]
        sample_indices = batch[-1]

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

        if debug and i == len(dataloader) - 1:
            batch_size, _, input_size, input_size = images.shape
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            # visualize_topk_prototypes(outputs, batch_im_paths, writer, step=epoch, input_size=input_size,
            #                           tag_fmt_str="Training first batch top{topk} prototypes/epoch {step}/{idx}")
            visualize_gt_class_prototypes(outputs, batch_im_paths, labels, writer, tag=f"Ground True Prototypes Train Epoch{epoch}", use_pooling=True)
            visualize_prototype_assignments(outputs, writer, step=epoch,
                                            tag=f"Training first batch prototype assignments/epoch {epoch}")

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
        images, labels  = batch[:2]
        sample_indices = batch[-1]

        outputs = model(images, labels=labels)

        mca_val(outputs["class_logits"], labels)

        if debug and i == len(dataloader) - 1:
            batch_size, input_size, input_size = images.shape
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            # visualize_topk_prototypes(outputs, batch_im_paths, writer, step=epoch, input_size=input_size,
            #                           tag_fmt_str="Validation epoch {step} batch 0 top{topk} prototypes/{idx}")
            visualize_gt_class_prototypes(outputs, batch_im_paths, labels, writer, tag=f"Ground True Prototypes Val Epoch{epoch}", use_pooling=True)
            visualize_prototype_assignments(outputs, labels, writer, step=epoch,
                                            tag=f"Validation epoch {epoch} batch {i} prototype assignments")

    epoch_acc_val = mca_val.compute().item()
    writer.add_scalar("Acc/val", epoch_acc_val, epoch)
    logger.info(f"EPOCH {epoch} val acc: {epoch_acc_val:.4f}")

    return epoch_acc_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir, resume_ckpt = setup_config_and_logging(name="train", base_log_dir="logs-23-09")

    logger = logging.getLogger(__name__)

    L.seed_everything(cfg.seed)

    normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        normalize
    ])

    n_classes, n_attributes = 200, 112
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_annotation_root = Path("datasets") / "class_attr_data_10"
    training_set_path = "train_cropped_augmented" if cfg.dataset.augmentation else "train_cropped"

    if cfg.get("few_shot", False):
        dataset_train_few_shot = CUBFewShotDataset((dataset_dir / "train_cropped").as_posix(),
                                                   n_samples_per_class=cfg.few_shot.get("n_samples", 10),
                                                   transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train_few_shot, batch_size=128, num_workers=8, shuffle=True)
    elif cfg.get("concept_learning", False):
        assert cfg.model.losses.l_attr_coef > 0
        dataset_train_concept = CUBConceptDataset((dataset_dir / training_set_path).as_posix(),
                                                  attribute_annotation_root=attribute_annotation_root,
                                                  transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train_concept, batch_size=128, num_workers=8, shuffle=True)
    else:
        dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                                   transforms=transforms)
        dataloader_train = DataLoader(dataset=dataset_train, batch_size=128, num_workers=8, shuffle=True)
    
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              transforms=transforms)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=8, shuffle=True)

    if "dino" in cfg.model.name:
        if cfg.model.n_splits and cfg.model.n_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=cfg.model.name,
                n_splits=cfg.model.n_splits,
                mode=cfg.model.tuning,
                freeze_norm_layer=cfg.model.get("freeze_norm", True)
            )
        else:
            backbone = DINOv2Backbone(name=cfg.model.name)
        dim = backbone.dim
    elif cfg.model.name.lower().startswith("clip"):
        backbone = MaskCLIP(name=cfg.model.name.split("-", 1)[1])
        dim = 512
    else:
        raise NotImplementedError("Backbone must be one of dinov2 or clip.")
    
    assert cfg.model.fg_extractor in ["PCA", "PaPr"]
    if cfg.model.fg_extractor == "PaPr":
        fg_extractor = PaPr(bg_class=n_classes, **cfg.model.fg_extractor_args)
    else:
        fg_extractor = PCA(bg_class=n_classes, **cfg.model.fg_extractor_args)

    net = ProtoDINO(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=cfg.model.n_prototypes,
        gamma=cfg.model.get("gamma", 0.99),
        temperature=cfg.model.temperature,
        cls_head="attribute" if cfg.get("concept_learning", False) else cfg.model.cls_head,
        sa_init=cfg.model.sa_init,
        n_attributes=n_attributes,
        norm_prototypes=cfg.model.get("norm_prototypes", False)
    )
    if resume_ckpt:
        state_dict = torch.load(resume_ckpt, map_location="cpu")
        net.load_state_dict(state_dict)

    criterion = ProtoPNetLoss(**cfg.model.losses, n_prototypes=cfg.model.n_prototypes)

    net.to(device)
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    best_epoch, best_val_acc = 0, 0.
    lr_coef = 1
    lr_decay = cfg.optim.get("lr_decay", 1)
    if (not lr_decay) or (lr_decay <= 0):
        lr_decay = 1

    for epoch in range(cfg.optim.epochs):
        is_fine_tuning = epoch in cfg.optim.fine_tuning_epochs
        is_debugging = epoch in cfg.debug.epochs

        if is_fine_tuning:
            logger.info("Start fine-tuning backbone...")
            for name, param in net.named_parameters():
                param.requires_grad = ("backbone" not in name) and ("fg_extractor" not in name)
            
            if cfg.model.tuning is not None:
                net.backbone.set_requires_grad()

            is_tuning_backbone = (cfg.model.n_splits != 0 and cfg.model.tuning is not None)
            param_groups = [{'params': net.backbone.learnable_parameters(),
                             'lr': cfg.optim.backbone_lr * lr_coef,
                             'weight_decay': cfg.optim.weight_decay if cfg.optim.get("weight_decay", False) else 0}] if is_tuning_backbone else []
            param_groups += [{'params': net.classifier.parameters(), 'lr': cfg.optim.cls_lr * lr_coef}]

            if cfg.optim.optimizer == "SGD":
                optimizer = optim.SGD(param_groups, momentum=0.9)
            if cfg.optim.optimizer == "Adam":
                optimizer = optim.Adam(param_groups)

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
        logger.info(f"lr_coef: {lr_coef}")

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

        if is_fine_tuning:
            lr_coef *= lr_coef

        epoch_acc_val = val_epoch(model=net, dataloader=dataloader_test, epoch=epoch,
                                  writer=writer, logger=logger, device=device)

        if epoch_acc_val > best_val_acc:
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, log_dir / "dino_v2_proto_best.pth")
            logger.info("Best epoch found, model saved!")
        elif epoch == (cfg.optim.epochs - 1):
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, log_dir / "dino_v2_proto_last_epoch.pth")
            logger.info("Last epoch, model saved!")

    logger.info(f"DONE! Best epoch is epoch {best_epoch} with accuracy {best_val_acc}.")


if __name__ == '__main__':
    main()
