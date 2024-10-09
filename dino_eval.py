#!/usr/bin/env python3
import logging
from logging import Logger
from math import sqrt
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, repeat
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from cub_dataset import CUBEvalDataset
from eval.concept_trustworthiness import (
    Cub2011Eval,
    evaluate_concept_trustworthiness,
    get_activation_maps,
)
from eval.consistency import evaluate_consistency
from eval.stability import evaluate_stability
from models.backbone import DINOv2Backbone, DINOv2BackboneExpanded, MaskCLIP
from models.dino import PCA, PaPr, ProtoDINO
from utils.config import load_config_and_logging
from utils.visualization import (
    visualize_prototype_assignments,
    visualize_prototype_part_keypoints,
    visualize_gt_class_prototypes,
)

N_KEYPOINTS = 15


@torch.no_grad()
def get_attn_maps(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, K: int = 5):
    patch_prototype_logits = model(images)["patch_prototype_logits"]  # type: torch.Tensor

    batch_size, n_patches, C, K = patch_prototype_logits.shape
    H = W = int(sqrt(n_patches))

    patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
    return patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W


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


def eval_nmi_ari(net: nn.Module, dataloader: DataLoader, device: str = "cpu"):
    """
    Get Normalized Mutual Information, Adjusted Rand Index for given method

    Parameters
    ----------
    net: nn.Module
        The trained net to evaluate
    data_loader: DataLoader
        The dataset to evaluate
    device: str

    Returns
    ----------
    nmi: float
        Normalized Mutual Information between predicted parts and gt parts as %
    ari: float
        Adjusted Rand Index between predicted parts and gt parts as %
    """
    device = torch.device(device)

    all_class_ids = []
    all_keypoint_part_assignments = []
    all_ground_truths = []

    for batch in tqdm(dataloader):
        batch = tuple(item.to(device) for item in batch)  # type: tuple[torch.Tensor, ...]
        images, keypoints, labels, attributes, sample_indices = batch
        batch_size, _, input_h, input_w = images.shape

        attn_maps = get_attn_maps(model=net, images=images, labels=labels)
        attn_maps_resized = F.interpolate(attn_maps, size=(input_h, input_w,), mode='bilinear', align_corners=False)

        kp_visibilities = (keypoints.sum(dim=-1) > 0).to(dtype=torch.bool)
        keypoints = keypoints.clone()[..., None, :]  # B, N_KEYPOINTS, 1, 2

        keypoints /= torch.tensor([input_w, input_h]).to(dtype=torch.float32, device=device)
        keypoints = keypoints * 2 - 1  # map keypoints from range [0, 1] to [-1, 1]

        keypoint_part_logits = F.grid_sample(attn_maps_resized, keypoints, mode='nearest', align_corners=False)  # B K N_KEYPOINTS, 1
        keypoint_part_assignments = torch.argmax(keypoint_part_logits, dim=1).squeeze()  # B N_KEYPOINTS

        for assignments, is_visible, class_id in zip(keypoint_part_assignments.unbind(dim=0), kp_visibilities.unbind(dim=0), labels):
            all_keypoint_part_assignments.append(assignments[is_visible])
            all_class_ids.append(torch.stack([class_id] * is_visible.sum()))
            all_ground_truths.append(torch.arange(N_KEYPOINTS, device=device)[is_visible])
    
    
    all_class_ids = torch.cat(all_class_ids, axis=0)
    all_keypoint_part_assignments_np = torch.cat(all_keypoint_part_assignments, axis=0)
    all_ground_truths = torch.cat(all_ground_truths, axis=0)

    all_classes_nmi, all_classes_ari = [], []

    for c in torch.unique(all_class_ids):
        mask = all_class_ids == c
        kp_part_assignment_c = all_keypoint_part_assignments_np[mask].flatten().cpu().numpy()
        ground_truths_c = all_ground_truths[mask].flatten().cpu().numpy()

        nmi = normalized_mutual_info_score(kp_part_assignment_c, ground_truths_c)
        ari = adjusted_rand_score(kp_part_assignment_c, ground_truths_c)

        all_classes_nmi.append(nmi)
        all_classes_ari.append(ari)

    return sum(all_classes_nmi) / len(all_classes_nmi), sum(all_classes_ari) / len(all_classes_nmi)


@torch.inference_mode()
def eval_consistency(net: nn.Module, dataloader: DataLoader, writer: SummaryWriter,
                     *,
                     K: int = 5, C: int = 200, N_PARTS: int = 15, H_b: int = 72, W_b: int = 72,
                     INPUT_SIZE: int = 224, threshold: float = 0.8, device: str | torch.device = "cpu"):
    """Deprecated in favour of original implementation"""
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
def eval_accuracy(model: nn.Module, dataloader: DataLoader, writer: SummaryWriter | None,
                  logger: Logger, device: torch.device, vis_every_n_batch: int = 5):
    model.eval()
    mca_eval = MulticlassAccuracy(num_classes=len(dataloader.dataset.classes), average="micro").to(device)

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(item.to(device) for item in batch)
        images, labels  = batch[:2]
        sample_indices = batch[-1]

        outputs = model(images, labels=labels)

        if i % vis_every_n_batch == 0 and writer is not None:
            batch_im_paths = [dataloader.dataset.samples[idx][0] for idx in sample_indices.tolist()]
            visualize_gt_class_prototypes(outputs, batch_im_paths, labels, writer, tag=f"Ground True Prototypes Eval Batch{i}", use_pooling=True)
            visualize_prototype_assignments(outputs, writer, step=i,
                                            tag=f"Eval prototype assignments/batch {i}")

        mca_eval(outputs["class_logits"], labels)

    epoch_acc_eval = mca_eval.compute().item()
    logger.info(f"Eval acc: {epoch_acc_eval:.4f}")

    return epoch_acc_eval

def push_forward(self: ProtoDINO, x: torch.Tensor):
    prototype_logits = self.__call__(x, labels=None)["patch_prototype_logits"]
    batch_size, n_patches, C, K = prototype_logits.shape
    H = W = int(sqrt(n_patches))
    prototype_logits = rearrange(prototype_logits, "B (H W) C K -> B (C K) H W", H=H, W=W)
    return None, F.avg_pool2d(prototype_logits, kernel_size=(2, 2,), stride=2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg, log_dir, args = load_config_and_logging(name="eval", return_args=True)

    logger = logging.getLogger(__name__)

    L.seed_everything(cfg.seed)
    
    n_classes, n_attributes = 200, 112
    dataset_dir = Path("datasets") / "cub200_cropped"
    annotations_path = Path("datasets") / "CUB_200_2011"

    dataset_eval = CUBEvalDataset((dataset_dir / "test_cropped").as_posix(), annotations_path.as_posix())
    dataloader_eval = DataLoader(dataset_eval, shuffle=True, batch_size=128)
    
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
    state_dict = torch.load(log_dir / "dino_v2_proto_last_epoch.pth", map_location="cpu")
    net.load_state_dict(state_dict=state_dict, strict=False)
    
    net.optimizing_prototypes = False
    if cfg.optim.epochs > 1:
        net.initializing = False
    net.eval()
    net.to(device)
    
    writer = SummaryWriter(log_dir=log_dir)

    logger.info("Evaluating accuracy...")
    eval_accuracy(model=net, dataloader=dataloader_eval, writer=writer, logger=logger, device=device, vis_every_n_batch=5)
    
    # logger.info("Evaluating class-wise NMI and ARI...")
    # mean_nmi, mean_ari = eval_nmi_ari(net=net, dataloader=dataloader_eval, device=device)
    # logger.info(f"Mean class-wise NMI: {float(mean_nmi)}")
    # logger.info(f"Mean class-wise ARI: {float(mean_ari)}")

    # Monkey-patch the model class to make it compatible with eval script
    ProtoDINO.push_forward = push_forward
    net.img_size = 224
    net.num_prototypes_per_class = net.n_prototypes

    args.data_path = "datasets"
    args.test_batch_size = 64
    args.nb_classes = 200

    logger.info("Evaluating consistency...")
    consistency_score = evaluate_consistency(net, args, save_dir=log_dir)
    logger.info(f"Network consistency score: {consistency_score.item()}")

    logger.info("Evaluating stability...")
    consistency_score = evaluate_stability(net, args)
    logger.info(f"Network stability score: {consistency_score.item()}")

    if cfg.get("concept_learning", False):
        logger.info("Evaluating concept trustworthiness...")
        normalize = T.Normalize(mean=(0.485, 0.456, 0.406,), std=(0.229, 0.224, 0.225,))

        transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            normalize,
            ])
        concept_loc_dataset_eval = Cub2011Eval(root='datasets/', train=False, transform=transform)
        concept_loc_dataloader_eval = DataLoader(concept_loc_dataset_eval, shuffle=False, batch_size=150)
        net.attributes_predictor = net.classifier[0]

        all_activation_maps, all_img_ids = get_activation_maps(net, concept_loc_dataloader_eval)
        mean_loc_acc, (all_loc_acc, all_attri_idx, all_num_samples) = evaluate_concept_trustworthiness(all_activation_maps, all_img_ids, bbox_half_size=45)
        logger.info(f"Concept trustworthiness score of the network on the {len(concept_loc_dataset_eval)} test images: {mean_loc_acc:.2f}%")
    if cfg.get("part_segmentation", False):
        logger.info("Evaluating part segmentation...")
        raise NotImplementedError


if __name__ == "__main__":
    main()
