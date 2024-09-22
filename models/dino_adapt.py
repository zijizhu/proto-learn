from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

from models.utils import momentum_update, sinkhorn_knopp


class PaPr(nn.Module):
    def __init__(self, backbone: str, q: float = 0.4, bg_class: int = 200):
        super().__init__()
        self.q = q
        if backbone.lower() == "resnet18":
            self.proposal = nn.Sequential(*list(resnet18(weights=ResNet18_Weights).children())[:-2])
        elif backbone.lower() == "mobilenet_v3_small":
            self.proposal = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights).features
        else:
            raise NotImplementedError

        self.bg_class = bg_class

    def forward(self, x: torch.Tensor, y: torch.Tensor, H: int = 16, W: int = 16):
        x = self.proposal(x)  # B dim h w
        x = x.mean(dim=1).unsqueeze(1)  # B 1 h w
        x = F.interpolate(x, size=(H, W,), mode="bicubic", align_corners=True)  # B 1 H W
        x = rearrange(x, "B D H W -> B (D H W)", D=1, H=H, W=W)
        
        quantiles = x.quantile(q=self.q, dim=-1, keepdim=True).to(device=x.device)  # B HW
        masks = torch.ge(x, other=quantiles)
        masks = rearrange(masks, "B (H W) -> B H W", H=H, W=W)

        pseudo_patch_labels = torch.where(masks, input=repeat(y, "B -> B H W", H=H, W=W), other=self.bg_class)

        return pseudo_patch_labels


class PCA(nn.Module):
    def __init__(self, compare_fn: str = "le", threshold: float = 0.5, n_components: int = 1, bg_class: int = 200) -> None:
        super().__init__()
        self.compare_fn = torch.ge if compare_fn == "ge" else torch.le
        self.threshold = threshold
        self.n_components = n_components
        self.bg_class = bg_class
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        B, n_patches, dim = x.shape
        H = W = int(sqrt(n_patches))
        U, _, _ = torch.pca_lowrank(
            x.reshape(-1, dim),
            q=self.n_components, center=True, niter=10
        )
        U_scaled = (U - U.min()) / (U.max() - U.min()).squeeze()  # shape: [B*H*W, 1]
        U_scaled = U_scaled.reshape(B, H, W)

        pseudo_patch_labels = torch.where(
            self.compare_fn(U_scaled, other=self.threshold),
            repeat(y, "B -> B H W", H=H, W=W),
            self.bg_class
        )

        return pseudo_patch_labels.to(dtype=torch.long)  # B H W


class ProtoDINO(nn.Module):
    def __init__(self, backbone: nn.Module, pooling_method: str, cls_head: str, fg_extractor: nn.Module,
                 *, adapter_type: str = "regular", gamma: float = 0.999, n_prototypes: int = 5, n_classes: int = 200,
                 temperature: float = 0.2, sa_init: float = 0.5, dim: int = 768):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        # self.C = n_classes
        self.backbone = backbone

        self.fg_extractor = fg_extractor

        self.feature_dim = dim
        self.dim = 64
        if adapter_type == "bottleneck":
            self.adapters = nn.ModuleDict(dict(
                feature=nn.Sequential(
                    nn.Linear(self.feature_dim, self.dim),
                    nn.ReLU(),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid()
                ),
                prototype=nn.Sequential(
                    nn.Linear(self.feature_dim, self.dim),
                    nn.ReLU(),
                    nn.Linear(self.dim, self.dim),
                    nn.Sigmoid()
                )
            ))
        else:
            self.adapters = nn.ModuleDict(dict(
                feature=nn.Sequential(
                    nn.Linear(self.feature_dim, self.dim),
                    nn.Sigmoid()
                ),
                prototype=nn.Sequential(
                    nn.Linear(self.feature_dim, self.dim),
                    nn.Sigmoid()
                )
            ))
        self.register_buffer("prototypes", torch.randn(self.C, self.n_prototypes, self.feature_dim))
        self.temperature = temperature

        assert pooling_method in ["sum", "avg", "max"]
        assert cls_head in ["fc", "sum", "avg", 'sa']
        self.pooling_method = pooling_method
        self.cls_head = cls_head

        self.sa = nn.Parameter(torch.full((self.n_classes, self.n_prototypes,), sa_init, dtype=torch.float32))

        # self.cls_fc = nn.Linear(self.feature_dim, self.n_classes)

        self.optimizing_prototypes = True
        self.initializing = True

    @staticmethod
    def online_clustering(prototypes: torch.Tensor,
                          patch_tokens: torch.Tensor,
                          patch_prototype_logits: torch.Tensor,
                          patch_labels: torch.Tensor,
                          bg_class: float | torch.Tensor,
                          *,
                          gamma: float = 0.999,
                          use_gumbel: bool = False):
        """Updates the prototypes based on the given inputs.
        This function updates the prototypes based on the patch tokens,
        patch-prototype logits, labels, and patch labels.

        Args:
            prototypes (torch.Tensor): A tensor of shape [C, K, dim,], representing K prototypes for each of C classes.
            patch_tokens: A tensor of shape [B, n_patches, dim,], which is the feature from backbone.
            patch_prototype_logits: The logits between patches and prototypes of shape [B, n_patches, C, K].
            patch_labels: A tensor of shape [B, H, W,] of type torch.long representing the (generated) patch-level class labels.
            labels: A tensor of shape [B,] of type torch.long representing the image-level class labels.
            gamma: A float indicating the coefficient for momentum update.
            use_gumbel: A boolean indicating whether to use gumbel softmax for patch assignments.
        """
        B, H, W = patch_labels.shape
        C, K, dim = prototypes.shape

        patch_labels_flat = patch_labels.flatten()  # shape: [B*H*W,]
        patches_flat = rearrange(patch_tokens, "B n_patches dim -> (B n_patches) dim")
        L = rearrange(patch_prototype_logits, "B n_patches C K -> (B n_patches) C K")

        P_old = prototypes.clone()
        P_new = prototypes.clone()

        part_assignment_maps = torch.empty_like(patch_labels_flat)
        L_c_dict = dict()  # type: dict[str, torch.Tensor]
        
        for c in patch_labels.unique().tolist():
            class_fg_mask = patch_labels_flat == c  # shape: [B*H*W,]
            I_c = patches_flat[class_fg_mask]  # shape: [N, dim]
            L_c = L[class_fg_mask, c, :]  # shape: [N, K,]

            L_c_assignment, L_c_assignment_indices = sinkhorn_knopp(L_c, use_gumbel=use_gumbel)  # shape: [N, K,], [N,]

            P_c_new = torch.mm(L_c_assignment.t(), I_c)  # shape: [K, dim]

            P_c_old = P_old[c, :, :]

            P_new[c, ...] = momentum_update(P_c_old, P_c_new, momentum=gamma)

            part_assignment_maps[class_fg_mask] = L_c_assignment_indices + c * K

            L_c_dict[c] = L_c_assignment

        part_assignment_maps = rearrange(part_assignment_maps, "(B H W) -> B (H W)", B=B, H=H, W=W)

        # P_new = F.normalize(P_new, p=2, dim=-1)

        return part_assignment_maps, P_new

    def get_fg_by_similarity(self, patch_prototype_logits: torch.Tensor, labels: torch.Tensor):
        batch_size, n_pathes, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(n_pathes))

        fg_logits = F.normalize(patch_prototype_logits[torch.arange(batch_size), :, labels, :].sum(dim=-1), p=1, dim=-1)
        bg_logits = F.normalize(patch_prototype_logits[torch.arange(batch_size), :, torch.full_like(labels, -1), :].sum(dim=-1), p=1, dim=-1)
        fg_logits, bg_logits = rearrange(fg_logits, "B (H W) -> B H W", H=H, W=W), rearrange(bg_logits, "B (H W) -> B H W", H=H, W=W)

        stacked = torch.stack([bg_logits, fg_logits], dim=-1)
        fg_mask = F.softmax(stacked, dim=-1).max(dim=-1).indices.to(torch.bool)  # B H W

        return torch.where(fg_mask, repeat(labels, "B -> B H W", H=H, W=W), C-1)

    def normalize_prototypes(self):
        self.learnable_prototypes.data = F.normalize(self.learnable_prototypes.data, p=2, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None,
                *, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)

        patch_tokens, cls_token = self.backbone(x)  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)

        if self.initializing:
            patch_prototype_logits = einsum(
                patch_tokens_norm,
                prototype_norm,
                "B n_patches dim, C K dim -> B n_patches C K"
            )
        else:
            patch_tokens_adapted_norm = self.adapters["feature"](patch_tokens)
            prototyp_adapted = self.adapters["prototype"](self.prototypes)

            patch_prototype_logits = einsum(
                patch_tokens_adapted_norm,
                F.normalize(prototyp_adapted, p=2, dim=-1),
                # F.normalize(self.learnable_prototypes, p=2, dim=-1),  # DEBUG
                "B n_patches dim, C K dim -> B n_patches C K"
            )

        image_prototype_logits, _ = patch_prototype_logits.max(1)  # shape: [B, C, K,], C=n_classes+1

        sa_weights = F.softmax(self.sa, dim=-1) * self.n_prototypes
        image_prototype_logits_weighted = image_prototype_logits[:, :-1, :] * sa_weights
        # image_prototype_logits_weighted = image_prototype_logits * sa_weights
        class_logits = image_prototype_logits_weighted.sum(-1)
        
        # aux_class_logits = self.cls_fc(cls_token)

        outputs = dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits,  # shape: [B, n_classes,]
            # aux_class_logits=aux_class_logits  # shape: B N_classes
        )

        if labels is not None:
            if self.initializing:
                pseudo_patch_labels = self.fg_extractor(patch_tokens_norm.detach(), labels)
            else:
                pseudo_patch_labels = self.get_fg_by_similarity(
                    patch_prototype_logits=patch_prototype_logits.detach(),
                    labels=labels
                )

            part_assignment_maps, new_prototypes = self.online_clustering(
                prototypes=self.prototypes,
                patch_tokens=patch_tokens_norm.detach(),
                patch_prototype_logits=patch_prototype_logits.detach(),
                patch_labels=pseudo_patch_labels,
                bg_class=self.C - 1,
                gamma=self.gamma,
                use_gumbel=use_gumbel
            )

            if self.training and self.optimizing_prototypes:
                self.prototypes = new_prototypes

            outputs.update(dict(
                patches=patch_tokens_norm,
                part_assignment_maps=part_assignment_maps,
                pseudo_patch_labels=pseudo_patch_labels
            ))

        return outputs


class ProtoPNetLoss(nn.Module):
    def __init__(
            self,
            l_orth_coef: float,
            l_clst_coef: float,
            l_patch_coef: float,
            l_sep_coef: float,
            l_aux_coef: float,
            num_classes: int = 200,
            n_prototypes: int = 5,
            temperature: float = 0.1,
            bg_class_weight: float = 0.1
        ) -> None:
        super().__init__()
        self.l_orth_coef = l_orth_coef
        self.l_clst_coef = l_clst_coef
        self.l_patch_coef = l_patch_coef
        self.l_sep_coef = l_sep_coef
        self.l_aux_coef = l_aux_coef
        self.xe = nn.CrossEntropyLoss()
        self.xe_aux = nn.CrossEntropyLoss()

        self.C = num_classes
        self.K = n_prototypes
        self.class_weights = torch.tensor([1] * self.C * self.K + [bg_class_weight] * self.K)
        self.temperature = temperature

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        # logits, aux_logits, similarities = outputs["class_logits"], outputs["aux_class_logits"], outputs["image_prototype_logits"]
        # patch_prototype_logits = outputs["patch_prototype_logits"]
        # features, part_assignment_maps, fg_masks = outputs["patches"], outputs["part_assignment_maps"], outputs["pseudo_patch_labels"]
        logits, similarities = outputs["class_logits"], outputs["image_prototype_logits"]
        _, labels, _ = batch

        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        # if self.l_aux_coef != 0:
        #     loss_dict["l_y_aux"] = self.xe_aux(aux_logits, labels)

        # if self.l_patch_coef != 0:
        #     l_patch = self.compute_patch_contrastive_cost(
        #         patch_prototype_logits,
        #         part_assignment_maps,
        #         class_weight=self.class_weights.to(dtype=torch.float32, device=logits.device),
        #         temperature=self.temperature
        #     )
        #     loss_dict["l_patch"] = self.l_patch_coef * l_patch
        #     loss_dict["_l_patch_raw"] = l_patch

        # if self.l_orth_coef != 0:
        #     l_orth = self.compute_orthogonality_costs(
        #         features=features,
        #         part_assignment_maps=part_assignment_maps,
        #         fg_masks=fg_masks,
        #         n_prototypes=self.K,
        #         bg_class_index=self.C
        #     )
        #     loss_dict["l_orth"] = self.l_orth_coef * l_orth
        #     loss_dict["_l_orth_raw"] = l_orth

        if self.l_clst_coef != 0 and self.l_sep_coef != 0:
            l_clst, l_sep = self.compute_prototype_costs(similarities, labels, num_classes=self.C + 1)
            loss_dict["l_clst"] = -(self.l_clst_coef * l_clst)
            loss_dict["l_sep"] = self.l_sep_coef * l_sep
            loss_dict["_l_clst_raw"] = l_clst
            loss_dict["_l_sep_raw"] = l_sep

        return loss_dict

    @staticmethod
    def compute_patch_contrastive_cost(patch_prototype_logits: torch.Tensor,
                                       patch_prototype_assignments: torch.Tensor,
                                       class_weight: torch.Tensor,
                                       temperature: float = 0.1):
        patch_prototype_logits = rearrange(patch_prototype_logits, "B N C K -> B (C K) N") / temperature
        loss = F.cross_entropy(patch_prototype_logits, target=patch_prototype_assignments, weight=class_weight)
        return loss
    
    @staticmethod
    def compute_orthogonality_costs(features: torch.Tensor,
                                    part_assignment_maps: torch.Tensor,
                                    fg_masks: torch.Tensor,
                                    n_prototypes: int,
                                    bg_class_index: int):
        fg_masks = rearrange(fg_masks, "B H W -> B (H W)")
        B, n_patches, dim = features.shape

        # Mask pooling
        part_assignment_maps = torch.where(fg_masks == bg_class_index, n_prototypes, part_assignment_maps)  # Set background patches to assignment value of n_prototypes
        assignment_masks = F.one_hot(part_assignment_maps, num_classes=n_prototypes + 1).to(dtype=torch.float32)  # B (HW) K
        mean_part_features = einsum(assignment_masks, features, "B HW K, B HW dim -> B K dim")
        
        mean_part_features = F.normalize(mean_part_features, p=2, dim=-1)

        cosine_similarities = torch.bmm(mean_part_features, rearrange(mean_part_features, "B K dim -> B dim K"))  # shape: B K dim, B dim K -> B K K
        cosine_similarities -= repeat(torch.eye(n_prototypes + 1, device=features.device), "H W -> B H W", B=B)

        return torch.sum(F.relu(torch.abs(cosine_similarities)))

    @staticmethod
    def compute_prototype_costs(similarities: torch.Tensor, labels: torch.Tensor, num_classes: int):
        positives = F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)
        negatives = 1 - positives
        cluster_cost = torch.mean(similarities.max(-1).values * positives)
        separation_cost = torch.mean(similarities.max(-1).values * negatives)

        return cluster_cost, separation_cost
