import torch
import torch.nn.functional as F
from torch import nn


class ProtoPConceptNet(nn.Module):
    def __init__(self, backbone: nn.Module, proj_layers: nn.Module, concept_vectors: torch.Tensor,
                 prototype_shape: tuple, init_weights=True, activation_fn='log'):
        super().__init__()
        self.prototype_shape = prototype_shape
        self.num_prototypes, self.dim, _, _ = prototype_shape
        self.num_classes, self.num_concepts = concept_vectors.shape
        self.epsilon = 1e-4

        assert activation_fn in ["log", "linear"]
        self.activation_fn = activation_fn

        assert (self.num_prototypes % self.num_classes == 0)
        self.register_buffer("proto_class_association", torch.zeros(self.num_prototypes, self.num_classes))
        num_proto_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.proto_class_association[j, j // num_proto_per_class] = 1

        self.backbone = backbone

        self.proj = proj_layers

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        self.register_buffer("ones", torch.ones(self.prototype_shape))

        self.fc_concepts = nn.Linear(self.num_prototypes, self.num_concepts, bias=False)
        self.register_buffer("fc_classes", torch.zeros(self.num_classes, self.num_prototypes))
        self.register_buffer("concept_vectors", concept_vectors)

        if init_weights:
            self._initialize_weights()

    def _l2_dists(self, x):
        """
        Compute x ** 2 - 2 * x * prototype + prototype ** 2
        All channels of x2_patch_sum at position i, j have the same values
        All spacial values of p2_reshape at each channel are the same
        """
        x2 = x ** 2  # shape: [b, c, h, w]
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)  # shape: [b, num_prototypes, h, w]

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))  # shape [num_prototypes, ]
        p2_reshape = p2.view(-1, 1, 1)  # shape [num_prototypes, 1, 1]

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # p2_reshape broadcasted to [b, num_prototypes, h, wv]

        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # shape: [b, num_prototypes, h, w]

    def distance_to_similarity(self, distances):
        if self.activation_fn == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances

    def forward(self, x, with_concepts=False):
        f = self.backbone(x)
        f = self.proj(f)
        dists = self._l2_dists(f)  # shape: [b, num_prototypes, h, w]
        b, num_prototypes, h, w = dists.shape
        # 2D min pooling
        min_dists = -F.max_pool2d(-dists, kernel_size=(h, w,))  # shape: [b, num_prototypes, 1, 1]
        min_dists = min_dists.squeeze((-1, -2,))  # shape: [b, num_prototypes]

        proto_activations = self.distance_to_similarity(min_dists)  # shape: [b, num_prototypes]

        if with_concepts:
            concept_logits = self.fc_concepts(proto_activations)
            class_logits = concept_logits @ self.concept_vectors.T
        else:
            concept_logits = None
            class_logits = proto_activations @ self.fc_classes.T
        return class_logits, concept_logits, min_dists, dists

    def _initialize_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        pos_connections = self.proto_class_association.T
        neg_connections = 1 - pos_connections
        self.fc_classes.data.copy_(1 * pos_connections - 0.5 * neg_connections)


class ProtoPNetLoss(nn.Module):
    def __init__(self, l_clst_coef: float, l_sep_coef: float):
        super().__init__()
        self.l_clst_coef = l_clst_coef
        self.l_sep_coef = l_sep_coef
        self.xe = nn.CrossEntropyLoss()

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                batch: dict[str, torch.Tensor],
                proto_class_association: torch.Tensor):
        class_logits, concept_logits, min_l2_dists, l2_dists = outputs
        _, labels, concept_labels = batch
        loss_dict = dict()
        loss_dict["l_y"] = self.xe(class_logits, labels)

        if self.l_clst_coef != 0 and self.l_sep_coef != 0:
            l_clst, l_sep = self.compute_costs(min_l2_dists,
                                               labels,
                                               proto_class_association)
            loss_dict["l_clst"] = self.l_clst_coef * l_clst
            loss_dict["l_sep"] = self.l_sep_coef * l_sep
        return loss_dict

    @staticmethod
    def compute_costs(l2_dists: torch.Tensor,
                      class_ids: torch.Tensor,
                      proto_class_association: torch.Tensor,
                      max_dist: int = 100):
        gt_proto_mask = proto_class_association[:, class_ids].T
        gt_proto_inverted_dists, _ = torch.max((max_dist - l2_dists) * gt_proto_mask, dim=1)
        cluster_cost = torch.mean(max_dist - gt_proto_inverted_dists)

        non_gt_proto_mask = 1 - gt_proto_mask
        non_gt_proto_inverted_dists, _ = torch.max((max_dist - l2_dists) * non_gt_proto_mask, dim=1)
        separation_cost = torch.mean(max_dist - non_gt_proto_inverted_dists)

        return cluster_cost, separation_cost
