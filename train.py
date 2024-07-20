#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from cub_dataset import CUBDataset
from models.backbone import load_backbone
from models.ppnet import ProtoPNet, ProtoPNetLoss, get_projection_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--options", "-o", type=str, nargs="+")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    log_dir = Path("logs")
    config = OmegaConf.load(Path(config_path))
    if args.options:
        log_dir /= "-".join(args.options)
        config.merge_with_dotlist(args.options)
    else:
        log_dir /= config_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(config))
    OmegaConf.save(config=config, f=log_dir / "config.yaml")

    L.seed_everything(42)
    transforms = T.Compose([
        T.Resize((224, 224,)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406,),
                    std=(0.229, 0.224, 0.225,))
    ])

    dataset_train = CUBDataset(Path("datasets") / "cub200_cropped" / "train_cropped_augmented",
                               Path("datasets") / "class_attribute_labels_continuous.txt",
                               transforms=transforms)
    dataset_test = CUBDataset(Path("datasets") / "cub200_cropped" / "test_cropped",
                              Path("datasets") / "class_attribute_labels_continuous.txt",
                              transforms=transforms)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=80, num_workers=8, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=100, num_workers=8, shuffle=False)

    num_classes = 200

    backbone, backbone_dim = load_backbone(config.model.backbone)
    proj_layers = get_projection_layer(config.model.proj_layers,
                                       first_dim=2048 if config.model.backbone == "resnet50" else 1024)
    
    ppnet = ProtoPNet(backbone, config.model.backbone, (2000, 128, 1, 1,), 200)
    criterion = ProtoPNetLoss(l_clst_coef=config.model.l_clst_coef,
                              l_sep_coef=config.model.l_sep_coef,
                              l_l1_coef=config.model.l_l1_coef)

    print("Projection Layers:")
    print(proj_layers)

    warmup_param_groups = [
        {'params': ppnet.proj.parameters(),
         'lr': config.optim.warmup.proj_lr,
         'weight_decay': config.optim.warmup.proj_weight_decay},
        {'params': ppnet.prototype_vectors,
         'lr': config.optim.warmup.proto_lr},
    ]
    joint_param_groups = [
        {'params': ppnet.backbone.parameters(),
         'lr': config.optim.joint.backbone_lr,
         'weight_decay': config.optim.joint.backbone_weight_decay},
        {'params': ppnet.proj.parameters(),
         'lr': config.optim.joint.proj_lr,
         'weight_decay': config.optim.joint.proj_weight_decay},
        {'params': ppnet.prototype_vectors,
         'lr': config.optim.joint.proto_lr}
    ]
    
    optimizer_warmup = optim.Adam(warmup_param_groups)
    optimizer_joint = optim.Adam(joint_param_groups)
    lr_scheduler_joint = optim.lr_scheduler.StepLR(optimizer_joint, step_size=5, gamma=0.1)

    if config.optim.final:
        final_param_groups = [
            {'params': ppnet.fc.parameters(),
            'lr': config.optim.final.fc_lr}
        ]
        optimizer_final = optim.Adam(final_param_groups)
    
    
    writer = SummaryWriter(log_dir=log_dir.as_posix())

    ppnet.train()
    ppnet.to(device)

    best_epoch, best_val_acc = 0, 0.
    early_stopping_epochs = 5

    optimizer = optimizer_warmup
    lr_scheduler = None
    for param in ppnet.backbone.parameters():
        param.requires_grad = False
    for param in ppnet.proj.parameters():
        param.requires_grad = True
    ppnet.prototype_vectors.requires_grad = True
    for param in ppnet.fc.parameters():
        param.requires_grad = False

    for epoch in range(25):
        if epoch == 5:
            for param in ppnet.backbone.parameters():
                param.requires_grad = True
            optimizer = optimizer_joint
            lr_scheduler = lr_scheduler_joint
        
        if config.optim.final and epoch == config.optim.final.start_epoch:
            for name, param in ppnet.named_parameters():
                param.requires_grad = bool("fc" in name)
            optimizer = optimizer_final
            lr_scheduler = None

        running_losses = defaultdict(float)
        mca_train = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)

        for batch in tqdm(dataloader_train):
            batch = tuple(item.to(device) for item in batch)
            images, labels, _ = batch
            outputs = ppnet(images)
            logits, dists = outputs
            loss_dict = criterion(outputs=outputs,
                                batch=batch,
                                proto_class_association=ppnet.proto_class_association,
                                fc_weights=ppnet.fc.weight)

            total_loss = sum(loss_dict.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in loss_dict.items():
                running_losses[k] += v.item() * dataloader_train.batch_size

            mca_train(logits, labels)

        for k, v in running_losses.items():
            loss_avg = v / len(dataloader_train.dataset)
            writer.add_scalar("Loss/apn/train", loss_avg, epoch)
            print(f"EPOCH {epoch} apn train {k}: {loss_avg:.4f}")
        epoch_acc_train = mca_train.compute().item()
        writer.add_scalar("Acc/apn/train", epoch_acc_train, epoch)
        print(f"EPOCH {epoch} apn train acc: {epoch_acc_train:.4f}")

        if lr_scheduler:
            lr_scheduler.step()

        del mca_train, outputs, loss_dict, running_losses

        # Validation loop
        mca_val = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
        with torch.no_grad():
            for batch in tqdm(dataloader_test):
                images, labels, _ = tuple(item.to(device) for item in batch)
                logits, dists = ppnet(images)

                mca_val(logits, labels)

        epoch_acc_val = mca_val.compute().item()
        writer.add_scalar("Acc/apn/val", epoch_acc_val, epoch)
        print(f"EPOCH {epoch} apn val acc: {epoch_acc_val:.4f}")

        # Early stopping based on validation accuracy
        if epoch_acc_val > best_val_acc:
            torch.save({k: v.cpu() for k, v in ppnet.state_dict().items()},
                        log_dir / f"ppnet_epoch{epoch}.pth")
            torch.save({k: v.cpu() for k, v in ppnet.state_dict().items()},
                        log_dir / "ppnet_best.pth")
            best_val_acc = epoch_acc_val
            best_epoch = epoch
            print("Best epoch found, model saved!")
        if epoch >= best_epoch + early_stopping_epochs:
            break