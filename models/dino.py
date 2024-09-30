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

class ScoreAggregation(nn.Module):
    def __init__(self, init_val: float = 0.2, n_classes: int = 200, n_prototypes: int = 5) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((n_classes, n_prototypes,), init_val, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor):
        n_classes, n_prototypes = self.weights.shape
        sa_weights = F.softmax(self.weights, dim=-1) * n_prototypes
        x = x * sa_weights  # B C K
        x = x.sum(-1)  # B C
        return x


class ProtoDINO(nn.Module):
    def __init__(self, backbone: nn.Module, fg_extractor: nn.Module, cls_head: str,
                 *, always_norm_patches: bool = True, gamma: float = 0.999, n_prototypes: int = 5, n_classes: int = 200,
                 norm_prototypes = False, temperature: float = 0.2, sa_init: float = 0.5, dim: int = 768, n_attributes: int = 112):
        super().__init__()
        self.gamma = gamma
        self.n_prototypes = n_prototypes
        self.n_classes = n_classes
        self.C = n_classes + 1
        self.backbone = backbone

        self.fg_extractor = fg_extractor

        self.dim = dim
        self.register_buffer("prototypes", torch.randn(self.C, self.n_prototypes, self.dim))
        self.temperature = temperature

        nn.init.trunc_normal_(self.prototypes, std=0.02)
        
        self.cls_head = cls_head
        if cls_head == "fc":
            self.classifier = nn.Linear(self.n_prototypes * self.n_classes, self.n_classes, bias=False)
            prototype_class_assiciation = torch.eye(self.n_classes).repeat_interleave(self.n_prototypes, dim=0)
            self.classifier.weight = nn.Parameter((prototype_class_assiciation - 0.5 * (1 - prototype_class_assiciation)).t())
        elif cls_head == "sa":
            self.classifier = ScoreAggregation(init_val=sa_init, n_classes=n_classes, n_prototypes=n_prototypes)
        elif cls_head == "attribute":
            self.classifier = nn.ModuleList([
                nn.Linear(self.n_prototypes * self.n_classes, n_attributes),
                nn.Linear(n_attributes, n_classes)
            ])
        else:
            raise NotImplementedError

        # self.aux_fc = nn.Linear(self.dim, self.n_classes)

        self.optimizing_prototypes = True
        self.initializing = True
        self.always_norm_patches = always_norm_patches
        self.norm_prototypes = norm_prototypes

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

            part_assignment_maps[class_fg_mask] = L_c_assignment_indices

            L_c_dict[c] = L_c_assignment

        part_assignment_maps = rearrange(part_assignment_maps, "(B H W) -> B (H W)", B=B, H=H, W=W)

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

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None,
                *, use_gumbel: bool = False):
        assert (not self.training) or (labels is not None)

        patch_tokens, cls_token = self.backbone(x)  # shape: [B, n_pathes, dim,]

        patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
        prototype_norm = F.normalize(self.prototypes, p=2, dim=-1)

        if not self.initializing:
            patch_prototype_logits = einsum(patch_tokens, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
            patch_prototype_similarities = einsum(patch_tokens_norm, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")
            image_prototype_similarities = patch_prototype_similarities.max(dim=1).values
        else:
            patch_prototype_logits = einsum(patch_tokens_norm, prototype_norm, "B n_patches dim, C K dim -> B n_patches C K")

        image_prototype_logits = patch_prototype_logits.max(1).values  # shape: [B, C, K,], C=n_classes+1

        if self.cls_head == "sa":
            class_logits = self.classifier(image_prototype_logits[:, :-1, :])
            class_logits = class_logits / self.temperature
            attribute_logits = None
        elif self.cls_head == "attribute":
            attribute_logits = self.classifier[0](image_prototype_logits[:, :-1, :].flatten(1, -1))
            class_logits = self.classifier[1](attribute_logits)
        else:
            raise NotImplementedError

        outputs = dict(
            patch_prototype_logits=patch_prototype_logits,  # shape: [B, n_patches, C, K,]
            image_prototype_logits=image_prototype_logits,  # shape: [B, C, K,]
            class_logits=class_logits,  # shape: [B, n_classes,]
            aux_class_logits=None,  # shape: B N_classes
            image_prototype_similarities=None if self.initializing else image_prototype_similarities,  # B C K
            attribute_logits=attribute_logits
        )

        if labels is not None:
            if self.initializing:
                if isinstance(self.fg_extractor, PaPr):
                    pseudo_patch_labels = self.fg_extractor(x, labels)
                else:
                    pseudo_patch_labels = self.fg_extractor(patch_tokens_norm.detach(), labels)
            else:
                pseudo_patch_labels = self.get_fg_by_similarity(
                    patch_prototype_logits=patch_prototype_logits.detach(),
                    labels=labels
                )
            pseudo_patch_labels = pseudo_patch_labels.detach()

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
                self.prototypes = F.normalize(new_prototypes, p=2, dim=-1) if self.norm_prototypes else new_prototypes

            outputs.update(dict(
                patches=patch_tokens_norm,
                part_assignment_maps=part_assignment_maps,
                pseudo_patch_labels=pseudo_patch_labels
            ))

        return outputs


class ProtoPNetLoss(nn.Module):
    def __init__(
            self,
            l_attr_coef: float = 0,
            l_dense_coef: float = 0,
            l_orth_coef: float = 0,
            l_clst_coef: float = 0,
            l_patch_coef: float = 0,
            l_sep_coef: float = 0,
            l_aux_coef: float = 0,
            num_classes: int = 200,
            n_prototypes: int = 5,
            temperature: float = 0.1,
            bg_class_weight: float = 0.1
        ) -> None:
        super().__init__()
        self.l_attr_coef = l_attr_coef
        self.l_dense_coef = l_dense_coef
        self.l_orth_coef = l_orth_coef
        self.l_clst_coef = l_clst_coef
        self.l_patch_coef = l_patch_coef
        self.l_sep_coef = l_sep_coef
        self.l_aux_coef = l_aux_coef
        self.xe = nn.CrossEntropyLoss()
        self.xe_aux = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.C = num_classes
        self.K = n_prototypes
        self.class_weights = torch.tensor([1] * self.C * self.K + [bg_class_weight] * self.K)
        self.temperature = temperature

    def forward(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]):
        logits, aux_logits, similarities = outputs["class_logits"], outputs["aux_class_logits"], outputs["image_prototype_similarities"]
        patch_prototype_logits = outputs["patch_prototype_logits"]
        features, part_assignment_maps, fg_masks = outputs["patches"], outputs["part_assignment_maps"], outputs["pseudo_patch_labels"]
        attribute_logits = outputs["attribute_logits"]
        _, labels, attributes, _ = batch

        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        if self.l_dense_coef != 0:
            l_dense = self.compute_dense_loss(
                patch_prototype_logits,
                patch_prototype_assignments=part_assignment_maps,
                labels=labels
            )
            loss_dict["l_dense"] = self.l_dense_coef * l_dense
            loss_dict["_l_dense_raw"] = l_dense

        if self.l_aux_coef != 0:
            l_y_aux = self.xe_aux(aux_logits, labels)
            loss_dict["l_aux"] = self.l_aux_coef * l_y_aux
            loss_dict["_l_aux_raw"] = loss_dict["l_y_aux"]

        if self.l_patch_coef != 0:
            l_patch = self.compute_patch_contrastive_cost(
                patch_prototype_logits,
                part_assignment_maps,
                class_weight=self.class_weights.to(dtype=torch.float32, device=logits.device),
                temperature=self.temperature
            )
            loss_dict["l_patch"] = self.l_patch_coef * l_patch
            loss_dict["_l_patch_raw"] = l_patch

        if self.l_orth_coef != 0:
            l_orth = self.compute_orthogonality_costs(
                features=features,
                part_assignment_maps=part_assignment_maps,
                fg_masks=fg_masks,
                n_prototypes=self.K,
                bg_class_index=self.C
            )
            loss_dict["l_orth"] = self.l_orth_coef * l_orth
            loss_dict["_l_orth_raw"] = l_orth

        if self.l_attr_coef != 0:
            assert attribute_logits is not None
            print(attribute_logits.dtype, attributes.dtype)
            l_attribute = self.bce(attribute_logits, attributes.to(torch.float32))
            loss_dict["l_attr"] = self.l_attr_coef * l_attribute
            loss_dict["_l_attr_raw"] = l_attribute

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
    def compute_dense_loss(patch_prototype_logits: torch.Tensor,
                           patch_prototype_assignments: torch.Tensor,
                           labels: torch.Tensor):
        """Supervise patch-level logits with assignment map"""
        B, N, C, K = patch_prototype_logits.shape
        H = W = int(sqrt(N))
        target_class_patch_prototype_logits = patch_prototype_logits[torch.arange(B), :, labels, :]  # shape: B N K
        target_class_patch_prototype_logits = rearrange(target_class_patch_prototype_logits, "B (H W) K -> B K H W", H=H, W=W)
        patch_prototype_assignments = rearrange(patch_prototype_assignments, "B (H W) -> B H W",  H=H, W=W)
        return F.cross_entropy(target_class_patch_prototype_logits, patch_prototype_assignments)
    
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
        positives = F.one_hot(labels, num_classes=num_classes).unsqueeze(dim=-1).to(dtype=torch.float32)
        negatives = 1 - positives

        cluster_cost = (similarities * positives).max(dim=-1).values.max(dim=-1).values
        separation_cost = (similarities * negatives).max(dim=-1).values.max(dim=-1).values

        return torch.mean(cluster_cost), torch.mean(separation_cost)
