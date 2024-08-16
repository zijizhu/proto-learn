import math
import re
from functools import partial

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from lightning.pytorch.cli import LightningCLI
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification.accuracy import Accuracy

from data import DataModule
from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import DinoVisionTransformer
from models.utils import get_cosine_schedule_with_warmup, block_expansion_dino


common_kwargs = dict(
    img_size=518,
    patch_size=14,
    mlp_ratio=4,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_fn=partial(Block, attn_class=MemEffAttention)
)

vit_small_kwargs = dict(embed_dim=384, num_heads=6)
vit_base_kwargs = dict(embed_dim=768, num_heads=12)

MODEL_DICT = {
    "dinov2_vits14_reg4": partial(DinoVisionTransformer, **vit_small_kwargs, **common_kwargs),
    "dinov2_vitb14_reg4": partial(DinoVisionTransformer, **vit_base_kwargs, **common_kwargs)
}

URL_DICT = {
    "dinov2_vits14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
}


class DINOFinetuning(L.LightningModule):
    def __init__(
        self,
        n_splits: int,
        model_name: str,
        *,
        n_classes: int = 200,
        loss: str = "soft_xe",
        training_mode: str = "block",
        
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        
        image_size: int = 224
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
        """
        super().__init__()
        self.save_hyperparameters()
        self.n_splits = n_splits
        self.model_name = model_name
        self.training_mode = training_mode
        self.loss = loss
        
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
    
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes

        self.image_size = image_size

        arch = MODEL_DICT[model_name]
        state_dict = torch.hub.load_state_dict_from_url(URL_DICT[model_name], map_location="cpu")
        if self.training_mode == "linear":
            self.net = arch(depth=12)
            self.net.load_state_dict(state_dict)
            for name, param in self.net.named_parameters():
                param.requires_grad = False
        elif self.training_mode == "block":
            expanded_state_dict, n_blocks, learnable_param_names = block_expansion_dino(
                state_dict=state_dict,
                n_splits=self.n_splits)
            self.net = arch(depth=n_blocks)
            self.net.load_state_dict(expanded_state_dict)

            for name, param in self.net.named_parameters():
                param.requires_grad = name in learnable_param_names
        else:
            raise ValueError(f"{self.training_mode} is not a valid mode. Use one of ['full', 'linear']")
        self.head = nn.Linear(self.net.embed_dim, self.n_classes)

        if self.loss == "soft_xe":
            self.loss_fn = SoftTargetCrossEntropy()
        elif self.loss == "xe":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError

        self.train_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        self.val_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        self.test_acc = Accuracy(num_classes=self.n_classes, task="multiclass", average="micro", top_k=1)
        
        if self.cutmix_alpha > 0 and self.mixup_alpha > 0:
            cutmix = v2.CutMix(alpha=self.cutmix_alpha, num_classes=self.n_classes)
            mixup = v2.MixUp(alpha=self.mixup_alpha, num_classes=self.n_classes)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup], p=[self.mix_prob, 1-self.mix_prob])
        else:
            self.cutmix_or_mixup = None
        
    def _check(self):
        self.print("Learnable parameters:")
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.print(name, "(zero-ed)" if param.detach().sum() == 0 else "")
    
    def on_train_start(self) -> None:
        self._check()

    def forward(self, x: torch.Tensor):
        feature_dict = self.net.forward_features(x)
        x = feature_dict["x_norm_clstoken"]
        x = self.head(x)
        return x

    def shared_step(self, batch: tuple[torch.Tensor, ...], mode: str = "train"):
        x, y = batch
        x = self(x)
        if self.loss == "soft_xe":
            y = F.one_hot(y, num_classes=self.n_classes) if (y.ndim == 1) else y
        loss = self.loss_fn(x, y)
        acc = getattr(self, f"{mode}_acc")(x, y)

        self.log(f"{mode}_loss", loss, on_epoch=True)
        self.log(f"{mode}_acc", acc, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        
        if self.cutmix_or_mixup is not None:
            batch = self.cutmix_or_mixup(*batch)

        return self.shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self.shared_step(batch, "val")

    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            num_warmup_steps=self.warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        
