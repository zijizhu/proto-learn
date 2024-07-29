import torch
from torch import nn

from torch.utils.data import DataLoader

from tqdm import tqdm


@torch.inference_mode()
def project_prototypes(net: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    """Updates prototype vectors with their nearest patches from training split"""
    num_proto, proto_dim, proto_h, proto_w = net.prototype_vectors.shape

    nearset_latent_patches = net.prototype_vectors.data.detach()
    nearset_patch_coords = torch.zeros((num_proto, 3,), device=device)
    min_l2_dists = torch.full((num_proto,), float('inf'), device=device)

    for batch in tqdm(dataloader):
        images, labels, concepts = tuple(item.to(device) for item in batch)
        outputs = net(images)
        
        for i in range(num_proto):
            proto_class_id, _ = i // 10, i % 10
            if proto_class_id not in labels:
                continue
            
            # Select samples from current batch that correspond to target class of prototype i
            target_class_mask = torch.eq(labels, proto_class_id)  # shape [batch_size,]
            # shape: [sum(labels == target_class), latent_h, latent_w]
            proto_i_batch_l2_dists = outputs["l2_dists"][target_class_mask, i, ...]
            # shape: [sum(labels == target_class), dim, latent_h, latent_w]
            target_class_features = outputs["projected_features"][target_class_mask, ...]
            
            # Check if there is a smaller distance for prototype i in this batch
            proto_i_batch_min_dist = torch.min(proto_i_batch_l2_dists)
            if min_l2_dists[i] < proto_i_batch_min_dist:
                continue
            
            # Found a smaller distance for prototype i. Update min distance for prototype i
            min_l2_dists[i] = proto_i_batch_min_dist
            # Update nearest patch for prototype i
            # TODO adapt to prototypes with spacial dimensions
            sample_index, patch_y, patch_x = torch.unravel_index(torch.argmin(proto_i_batch_l2_dists),
                                                                proto_i_batch_l2_dists.shape)
            nearset_latent_patches[i, :, 0, 0] = target_class_features[sample_index, :, patch_y, patch_x]

    net.prototype_vectors.data.copy_(nearset_latent_patches)
    
    return dict(
        nearset_patch_coords=nearset_patch_coords.detach().cpu(),
        min_l2_dists=min_l2_dists.detach().cpu()
    )
