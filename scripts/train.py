#!/usr/bin/env python3
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "dinov2"))
import logging
from pathlib import Path

import lightning as L
import torch
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from models.dino import ProtoDINO, ProtoPNetLoss
from models.backbone import DINOv2BackboneExpanded, load_adapter
from cub_dataset import CUBDataset
from utils.visualization import visualize_prototype_assignments, visualize_topk_prototypes
from utils.config import setup_config_and_logging
from models.utils import get_cosine_schedule_with_warmup, print_parameters


class DINOLightningModule(L.LightningModule):
    def __init__(
        self,
        mode: str,
        backbone_name: str,
        final_layer: str,
        l_clst_coef: float,
        l_sep_coef: float,
        l_l1_coef: float,
        backbone_lr: float,
        final_layer_lr: float,
        *,
        adapter_config: str = "",
        n_splits: int = 3,
        input_size: int = 224,
        debugging_batch_idx: int = 0):
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.final_layer = final_layer

        self.backbone_lr = backbone_lr
        self.final_layer_lr = final_layer_lr

        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))
        self.transforms = T.Compose([
            T.Resize((input_size, input_size,)),
            T.ToTensor(),
            self.normalize
        ])

        dataset_dir = Path("datasets") / "cub200_cropped"
        attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
        training_set_path = "train_cropped_augmented" if config["dataset"]["augmentation"] else "train_cropped"
        self.dataset_train = CUBDataset((dataset_dir / training_set_path).as_posix(),
                                   attribute_labels_path.as_posix(),
                                   transforms=self.transforms)
        self.dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                                  attribute_labels_path.as_posix(),
                                  transforms=self.transforms)

        adapter = load_adapter(config["model"]["adapter_config"])
        if config["model"]["adapter_config"]:
            print(adapter)
        self.backbone = DINOv2BackboneExpanded(name=config["model"]["name"], n_splits=config["model"]["n_splits"])
        self.head = ProtoDINO(backbone=backbone,
                        adapter=adapter,
                        pooling_method=config["model"]["pooling_method"],
                        cls_head=config["model"]["cls_head"],
                        dim=backbone.dim,
                        pca_fg_cmp=config["model"]["pca_fg_cmp"]
        )

        for params in self.net.parameters():
            params.requires_grad = False

        self.criterion = ProtoPNetLoss(l_clst_coef=config["model"]["losses"]["l_clst_coef"],
                                  l_sep_coef=config["model"]["losses"]["l_sep_coef"],
                                  l_l1_coef=config["model"]["losses"]["l_l1_coef"])
        
        self.train_accuracy = MulticlassAccuracy(num_classes=len(self.dataset_train.classes), average="micro")
        self.val_accuracy = MulticlassAccuracy(num_classes=len(self.dataset_test.classes), average="micro")

        self.best_val_acc = 0.
        self.best_epoch = 0
        
        self.debugging_batch_idx = debugging_batch_idx

    @property()
    def optimizing_prototypes(self):
        return self.training and self.current_epoch % 2 == 0
    
    @property()
    def optimizing_backbone(self):
        return self.training and not self.optimizing_prototypes
    
    @property()
    def debugging(self):
        return self.current_epoch in self.debugging_epochs
    
    def on_train_epoch_start(self):
        if not self.optimizing_backbone:
            return
        if self.mode == "block_expansion":
            self.backbone.set_requires_grad()
        else:
            for param in self.head.parameters():
                param.requires_grad = True

        print_parameters(net=self.backbone, logger=self.logger)
        print_parameters(net=self.head, logger=self.logger)
    
    def setup(self, stage: str) -> None:
        self.sample_paths = self.trainer.train_dataloader.dataset.samples

    def forward(self, x, labels=None):
        x = self.backbone(x)
        return self.net(x, labels=labels)
    
    def _shared_step(self, batch, batch_idx, name: str = "train"):
        images, labels, _, sample_indices = batch
        outputs = self(images, labels=labels, debug=False, use_gumbel=False)

        if self.optimizing_backbone:
            loss_dict = self.criterion(outputs, batch)
            loss = sum(val for key, val in loss_dict.items() if not key.startswith("_"))
            self.log_dict({f"train_loss/{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True)
        else:
            loss = None

        if self.debugging and batch_idx == self.debugging_batch_idx:
            batch_im_paths = [self.dataset_train.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, self.logger.experiment, self.current_epoch, epoch_name=name)
            visualize_prototype_assignments(outputs, labels, self.logger.experiment, self.current_epoch, epoch_name=name, figsize=(8, 10,))
            
        return outputs, loss

    def training_step(self, batch, batch_idx):
        images, labels, _, sample_indices = batch
        outputs, loss = self._shared_step(batch=batch, batch_idx=batch_idx, name="train")
        self.train_accuracy(outputs["class_logits"], labels)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _, sample_indices = batch
        outputs, loss = self._shared_step(batch=batch, batch_idx=batch_idx, name="val")
        self.val_accuracy(outputs["class_logits"], labels)
        self.log("val_loss", self.val_accuracy, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        param_groups = []
        if self.mode == "block_expansion":
            param_groups += [{'params': filter(lambda p: p.requires_grad, self.backbone.parameters()), 'lr': self.backbone_lr}]
        param_groups += [{'params': self.head.sa, 'lr': self.final_layer_lr}] if self.final_layer == "sa" else []
        param_groups += [{'params': self.head.fc, 'lr': self.final_layer_lr}] if self.final_layer == "fc" else []

        optimizer = optim.SGD(param_groups, momentum=0.9)
        return optimizer


def main():
    config, log_dir = setup_config_and_logging(base_log_dir="logs_new")
    logger = logging.getLogger(__name__)

    L.seed_everything(42)

    model = DINOLightningModule(config)

    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=config["optim"]["epochs"],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=L.loggers.TensorBoardLogger(save_dir=log_dir),
    )

    trainer.fit(model)

    logger.info(f"DONE! Best epoch is epoch {model.best_epoch} with accuracy {model.best_val_acc}.")


if __name__ == '__main__':
    main()