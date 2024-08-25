#!/usr/bin/env python3
import os
import sys
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "dinov2"))
import logging
from logging import Logger
from math import sqrt
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from cub_dataset import CUBEvalDataset
from models.backbone import DINOv2BackboneExpanded
from models.dino import ProtoDINO
from utils.config import load_config_and_logging
from utils.visualization import (
    visualize_prototype_assignments,
    visualize_topk_prototypes,
    visualize_prototype_part_keypoints
)


def in_bbox(keypoint: tuple[float, float], bbox: tuple[float, float, float, float]):
    kp_x, kp_y = keypoint
    x, y, w, h = bbox
    return (x <= kp_x <= x + w) and (y <= kp_y <= y + h)


def compute_part_vectors(activation_maps: torch.Tensor, keypoints: torch.Tensor, height=72, width=72, input_size: int = 224):
    kp_visibility = (keypoints.sum(dim=-1) > 0).to(dtype=torch.long)
    part_vectors = []
    bboxes = []
    K, H, W = activation_maps.shape
    for act_map in activation_maps:
        cy, cx = torch.unravel_index(torch.argmax(act_map), (H, W,))
        cy, cx = cy.item(), cx.item()
        bbox_coord = (min(max(cx - width // 2, 0), input_size), min(max(cy - height // 2, 0), input_size), width, height,)
        kp_in_part = torch.tensor([in_bbox(kp, bbox_coord) for kp in keypoints.tolist()], dtype=torch.long, device=activation_maps.device)
        part_vectors.append(kp_in_part)
        bboxes.append(bbox_coord)
        
    return torch.stack(part_vectors), kp_visibility, bboxes


@torch.inference_mode()
def eval_consistency(net: nn.Module, dataloader: DataLoader, writer: SummaryWriter,
                     *,
                     K: int = 5, C: int = 200, N_PARTS: int = 15, H_b: int = 72, W_b: int = 72,
                     INPUT_SIZE: int = 224, threshold: float = 0.8, device: str | torch.device = "cpu"):
    net.eval()
    O_p, O_sum = torch.zeros((C, K, N_PARTS,), dtype=torch.long, device=device), torch.zeros((C, K, N_PARTS,), dtype=torch.long, device=device)

    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)
        transformed_im, transformed_keypoints, labels, attributes, sample_indices = batch
        outputs = net(transformed_im)

        for i, (activations, keypoints, c) in enumerate(zip(outputs["patch_prototype_logits"], transformed_keypoints, labels)):
            H = W = int(sqrt(activations.size(0)))
            activations = rearrange(activations, "(H W) C K -> C K H W", H=H, W=W)
            activations = F.interpolate(activations, INPUT_SIZE, mode="bicubic")  # shape: [C, K, INPUT, INPUT,]
            activations_c = activations[c, ...]  # shape: [K, H, W,]
            part_vectors, part_visibilities, bboxes = compute_part_vectors(activation_maps=activations_c, keypoints=keypoints, height=H_b, width=W_b)  # shape: [K, N_PARTS,], [N_PARTS,]

            O_p[c] += part_vectors
            O_sum[c] += repeat(part_visibilities, "n_parts -> K n_parts", K=K)
            
            if i == 0:
                batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
                visualize_prototype_part_keypoints(
                    im_path=batch_im_paths[i],
                    activation_maps=activations_c,
                    keypoints=keypoints,
                    part_vector=part_vectors,
                    part_visibility=part_visibilities,
                    bboxes=bboxes,
                    sample_id=sample_indices[i].item(),
                    writer=writer
                )

    a_p = O_p / O_sum
    consistencies = (a_p.max(-1).values >= threshold).to(torch.float32)
    return torch.mean(consistencies)


@torch.inference_mode()
def eval_accuracy(model: nn.Module, dataloader: DataLoader, writer: SummaryWriter,
                  logger: Logger, device: torch.device, vis_every_n_batch: int = 5):
    model.eval()
    mca_eval = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, transformed_keypoints, labels, attributes, sample_indices = batch
        outputs = model(images, labels=labels)

        if i % vis_every_n_batch == 0:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_topk_prototypes(outputs, batch_im_paths, writer, i, epoch_name="eval")
            visualize_prototype_assignments(outputs, labels, writer, i, epoch_name="eval", figsize=(10, 10,))

        mca_eval(outputs["class_logits"], labels)

    epoch_acc_eval = mca_eval.compute().item()
    logger.info(f"Eval acc: {epoch_acc_eval:.4f}")

    return epoch_acc_eval


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir = load_config_and_logging(name="eval")

    logger = logging.getLogger(__name__)

    L.seed_everything(42)
    
    dataset_dir = Path("datasets") / "cub200_cropped"
    annotations_path = Path("datasets") / "CUB_200_2011"

    dataset_eval = CUBEvalDataset((dataset_dir / "test_cropped").as_posix(), annotations_path.as_posix())
    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=128)
    
    backbone = DINOv2BackboneExpanded(name=cfg.model.name, n_splits=cfg.model.n_splits)
    net = ProtoDINO(
        backbone=backbone,
        dim=backbone.dim,
        learn_scale=cfg.model.learn_scale,
        pooling_method=cfg.model.pooling_method,
        cls_head=cfg.model.cls_head,
        pca_compare=cfg.model.pca_compare
    )
    state_dict = torch.load(log_dir / "dino_v2_proto.pth", map_location="cpu")
    net.load_state_dict(state_dict=state_dict)
    
    net.optimizing_prototypes = False
    net.eval()
    net.to(device)
    
    writer = SummaryWriter(log_dir=log_dir)
    logger.info("Evaluating accuracy...")
    eval_accuracy(model=net, dataloader=dataloader_eval, writer=writer, logger=logger, device=device, vis_every_n_batch=5)

    logger.info("Evaluating consistency...")
    consistency_score = eval_consistency(net, dataloader_eval, writer=writer, device=device)
    logger.info(f"Network consistency score: {consistency_score.item()}")


if __name__ == "__main__":
    main()
    