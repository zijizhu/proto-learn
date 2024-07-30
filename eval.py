import argparse
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes, make_grid
from tqdm import tqdm

from models.backbone import load_backbone
from cub_dataset import CUBDataset
from models.pp_concept_net import ProtoPConceptNet


def patch_coord_to_bbox(coord: tuple[int, int], input_size=224, latent_size=7):
    patch_size = input_size // latent_size
    latent_y, latent_x = coord
    return latent_y * patch_size, latent_x * patch_size, patch_size, patch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    with_concepts = "concept" in log_dir.as_posix()

    config = OmegaConf.load(log_dir / "config.yaml")

    print(OmegaConf.to_yaml(config))
    L.seed_everything(config.optim.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406,),
                    std=(0.229, 0.224, 0.225,))
    ])
    dataset_dir = Path("datasets") / "cub200_cropped"
    attribute_labels_path = Path("datasets") / "class_attribute_labels_continuous.txt"
    dataset_train = CUBDataset((dataset_dir / "train_cropped").as_posix(),
                               attribute_labels_path.as_posix(),
                               transforms=transforms)
    dataset_test = CUBDataset((dataset_dir / "test_cropped").as_posix(),
                              attribute_labels_path.as_posix(),
                              transforms=transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=8, num_workers=8, shuffle=False)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, num_workers=8, shuffle=False)

    
    if with_concepts:
        from models.pp_concept_net import ProtoPNet, get_projection_layer
        backbone, dim = load_backbone(backbone_name=config.model.backbone)
        proj_layers = get_projection_layer(config.model.proj_layers, first_dim=dim)
        ppnet = ProtoPConceptNet(backbone, proj_layers, dataset_train.attributes, tuple(config.model.prototype_shape))
    elif "dino" in config.model.backbone:
        from models.ppnet_dino import ProtoPNetDINO, get_projection_layer
        proj_layers = get_projection_layer(config.model.proj_layers)
    
        ppnet = ProtoPNetDINO("dinov2_vitb14_reg", proj_layers=proj_layers,
                              prototype_shape=tuple(config.model.prototype_shape), num_classes=200)
    else:
        from models.ppnet import ProtoPNet, get_projection_layer
        ppnet = ProtoPNet(backbone, proj_layers, tuple(config.model.prototype_shape), 200)

    state_dict = torch.load(log_dir / "ppnet_best.pth")
    ppnet.load_state_dict(state_dict)

    writer = SummaryWriter(log_dir.as_posix())

    ppnet.to(device)
    ppnet.eval()

    if args.eval:
        print("Start evaluating accuracy on test set..")
        # Inference on test set
        mca = MulticlassAccuracy(num_classes=200, average="micro").to(device)
        with torch.inference_mode():
            for batch in tqdm(dataloader_test):
                batch = tuple(item.to(device) for item in batch)
                images, labels, _ = batch
                outputs = ppnet(images)
                mca(outputs["logits"], labels)
        test_acc = mca.compute().item()
        writer.add_scalar("Acc/Test", test_acc)
        print(f"Test Accuracy: {test_acc:.4f}%")

    # Inference on training set to get exemplar patches for each prototype
    all_sample_proto_dists, all_logits = [], []
    with torch.inference_mode():
        for sample in tqdm(dataloader_train):
            sample = tuple(item.to(device) for item in sample)
            sample_image, label, _ = sample
            outputs = ppnet(sample_image)
            
            logits, dists = outputs["logits"], outputs["l2_dists"]

            all_sample_proto_dists.append(dists)
            all_logits.append(logits)

    # Compute top k patches from training set that has smallest l2 distance to each prototype
    train_proto_dists = torch.cat(all_sample_proto_dists).cpu()
    num_samples, num_proto, w, h = train_proto_dists.shape
    train_proto_dists_flat = train_proto_dists.permute(1, 0, 2, 3).reshape(num_proto, num_samples * w * h)
    proto_min_dists, indices = (-train_proto_dists_flat).topk(dim=-1, k=5)
    proto_min_dists = -proto_min_dists
    indices = torch.unravel_index(indices, (num_samples, w, h))

    # shape: [num_proto, topk, 3]; 3 columns are sample index, w, h at last dimension
    patch_coords = torch.stack(indices, dim=-1)
    torch.save(patch_coords, log_dir / "train_proto_nearest_patches.pth")
    torch.save(train_proto_dists, log_dir / "train_proto_dists.pth")

    # shape: [num_proto, topk, patch_h, patch_w], [num_proto, topk, input_h, input_w]
    proto_to_patches = [[None] * 5 for _ in range(num_proto)]
    proto_to_src_images = [[None] * 5 for _ in range(num_proto)]

    for i, sample in enumerate(tqdm(dataset_train.samples)):
        im_path, label = sample
        mask = torch.eq(patch_coords[:, :, 0], i)  # shape: [num_prototypes, topk]
        proto_patch_indices = torch.nonzero(mask)
        if proto_patch_indices.numel() == 0:
            continue
        proto_indices, topk_indices = proto_patch_indices.unbind(dim=-1)
        coords = patch_coords[proto_indices, topk_indices, :]

        im_pt = read_image(im_path, mode=ImageReadMode.RGB)
        im_pt = F.resize(im_pt, size=[224, 224])

        for proto_idx, topk_idx, c in zip(proto_indices, topk_indices, coords.tolist()):
            _, patch_h, patch_w = c
            y, x, w, h = patch_coord_to_bbox((patch_h, patch_w,))
            patch_cropped = im_pt[:, y:y + h, x:x + w]
            src_im_with_bbox = draw_bounding_boxes(im_pt, boxes=torch.tensor([[x, y, x + w, y + h]]),
                                                   colors="red", width=2)
            proto_to_patches[proto_idx][topk_idx] = patch_cropped
            proto_to_src_images[proto_idx][topk_idx] = src_im_with_bbox

    all_patch_grids, all_src_im_grids = [], []
    for i, (patches, src_imgs) in enumerate(tqdm(zip(proto_to_patches, proto_to_src_images), total=num_proto)):
        patches_grid = make_grid(patches)
        src_im_grid = make_grid(src_imgs)
        writer.add_image(f"Prototype_Patches_Top5/{i}", patches_grid)
        writer.add_image(f"Prototype_Src_Images_Top5/{i}", src_im_grid)
        all_patch_grids.append(patches_grid)
        all_src_im_grids.append(src_im_grid)


if __name__ == '__main__':
    main()
