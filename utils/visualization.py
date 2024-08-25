from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torchvision.transforms.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def overlay_attn_map(attn_map: np.ndarray, im: Image.Image):
    """
    attn_map: np.ndarray of shape (H, W,), same as im, values can be unormalized
    im: a PIL image of width H and height W
    """
    im = np.array(im.resize((224, 224), Image.BILINEAR))

    max_val, min_val = np.max(attn_map), np.min(attn_map)
    attn_map = (attn_map - min_val) / (max_val - min_val)
    heatmap = cv2.applyColorMap((255 * attn_map).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return (0.5 * heatmap + 0.5 * im).astype(np.uint8)


def visualize_topk_prototypes(batch_outputs: dict[str, torch.Tensor],
                              batch_im_paths: list[str],
                              writer: SummaryWriter,
                              epoch: int,
                              epoch_name: str,
                              *,
                              topk: int = 5,
                              input_size: int = 224,
                              latent_size: int = 16):
    batch_size, C, K = batch_outputs["image_prototype_logits"].shape
    b = 0
    H = W = latent_size

    batch_prototype_logits = rearrange(batch_outputs["image_prototype_logits"].detach(), "B C K -> B (C K)")
    batch_saliency_maps = rearrange(batch_outputs["patch_prototype_logits"].detach(), "B (H W) C K -> B H W C K", H=H, W=W)
    
    figures = []
    for b, (prototype_logits, saliency_maps, im_path) in enumerate(zip(batch_prototype_logits, batch_saliency_maps, batch_im_paths)):

        logits, indices = prototype_logits.topk(k=topk, dim=-1)
        indices_C, indices_K = torch.unravel_index(indices=indices, shape=(C, K,))
        topk_maps = saliency_maps[:, :, indices_C, indices_K]
        
        overlayed_images = []
        src_im = Image.open(im_path).convert("RGB").resize((input_size, input_size,))

        topk_maps_resized_np = cv2.resize(topk_maps.cpu().numpy(),
                                          (input_size, input_size,),
                                          interpolation=cv2.INTER_LINEAR)
        
        overlayed_images = [overlay_attn_map(topk_maps_resized_np[:, :, i], src_im) for i in range(topk)]  # shaspe: [topk, input_size, input_size, 3]
        
        fig, axes = plt.subplots(1, topk, figsize=(topk+2, 2))
        for ax, im, c, k in zip(axes.flat, overlayed_images, indices_C.tolist(), indices_K.tolist()):
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{c}/{k}")

        class_name, fname = Path(im_path).parts[-2:]
        fig.suptitle(f"{class_name}/{fname} top {topk} prototypes")
        fig.tight_layout()
        
        fig.canvas.draw()
        fig_image = Image.frombuffer('RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig=fig)
        
        figures.append(fig_image)
        writer.add_image(f"Epoch {epoch} top {topk} prototypes/{epoch_name}/{b}", F.pil_to_tensor(fig_image))

    return figures


def visualize_prototype_assignments(outputs: dict[str, torch.Tensor], labels: torch.Tensor, writer: SummaryWriter,
                                    step: int, epoch_name: str, figsize: tuple[int, int] = (8, 10,)):
    patch_labels = outputs["pseudo_patch_labels"].detach().clone()  # shape: [B, H, W,]
    L_c_dict = {c: L_c.detach().clone() for c, L_c in outputs["L_c_dict"].items()}

    nrows, ncols = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10,))

    for b, (c, ax) in enumerate(zip(labels.cpu().tolist(), axes.flat)):
        patch_labels_b = patch_labels[b, :, :]  # shape: [H, W,], dtype: torch.long
        fg_mask_b = patch_labels_b != 200  # shape: [H, W,], dtype: bool

        num_foreground_pixels = fg_mask_b.sum().cpu().item()

        L_c_i = L_c_dict[c][:num_foreground_pixels]
        L_c_dict[c] = L_c_dict[c][num_foreground_pixels:]
        L_c_i_argmax = L_c_i.argmax(-1)  # shape: [N,]

        assignment_map = torch.empty_like(fg_mask_b, dtype=torch.long)
        assignment_map[fg_mask_b] = L_c_i_argmax
        assignment_map[~fg_mask_b] = -1

        ax.imshow((assignment_map + 1).squeeze().cpu().numpy(), cmap="tab10")
        ax.set_xticks([]), ax.set_yticks([])
    
    fig.tight_layout()
    fig.canvas.draw()
    fig_image = Image.frombuffer('RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig=fig)
    
    writer.add_image(f"Batch prototype assignment/{epoch_name}", F.pil_to_tensor(fig_image), global_step=step)
    
    return fig_image


def visualize_prototype_part_keypoints(im_path: str, activation_maps: torch.Tensor, keypoints: list[tuple[float, float]],
                                       part_vector: torch.Tensor, part_visibility: torch.Tensor,
                                       bboxes: tuple[int, ...], sample_id: int, writer: SummaryWriter, input_size: int = 224):
    K, height, width = activation_maps.shape
    N_PARTS = len(keypoints)
    assert len(bboxes) == K

    src_im = Image.open(im_path).convert("RGB").resize((input_size, input_size,))
    visible_keypoints = np.array(list(kp for kp in keypoints if sum(kp) > 0))
    kp_x, kp_y = visible_keypoints[:, 0], visible_keypoints[:, 1]
    part_vector_np, part_visibility_np = part_vector.cpu().numpy(), part_visibility.cpu().numpy()

    fig, axes = plt.subplots(3, K, figsize=(8, 8))
    for i in range(K):
        im_overlayed = overlay_attn_map(activation_maps[i].cpu().numpy(), src_im)
        axes[0, i].imshow(im_overlayed)
        axes[0, i].set_xticks([]), axes[0, i].set_yticks([])

        axes[1, i].imshow(src_im)
        axes[1, i].plot(kp_x, kp_y, "ro")

        x, y, w, h = bboxes[i]
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1, i].add_patch(rect)
        axes[1, i].set_xticks([]), axes[1, i].set_yticks([])
        
        arr = np.vstack([part_vector_np[i], part_visibility_np])
        axes[2, i].matshow(1 - arr, cmap='Pastel1')
        for (i, j), z in np.ndenumerate(arr):
            axes[2, i].text(j, i, z, ha='center', va='center')
            axes[2, i].set_xticks(range(N_PARTS))
            axes[2, i].set_yticks([])
    
    fig.tight_layout()
    fig.canvas.draw()
    fig_image = Image.frombuffer('RGBa', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig=fig)
    
    writer.add_image(f"Prototype Part Keypoints/{sample_id}", F.pil_to_tensor(fig_image))
    return fig_image
