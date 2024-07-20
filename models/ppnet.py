import torch
import torch.nn.functional as F
from torch import nn


class ProtoPNet(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_name: str, prototype_shape, num_classes,
                 init_weights=True, activation_fn='log'):
        super().__init__()
        self.prototype_shape = prototype_shape
        self.num_prototypes, self.dim, _, _ = prototype_shape
        self.num_classes = num_classes
        self.epsilon = 1e-4

        assert activation_fn in ["log", "linear"]
        self.activation_fn = activation_fn

        assert(self.num_prototypes % self.num_classes == 0)
        self.register_buffer("proto_class_association", torch.zeros(self.num_prototypes, self.num_classes))
        num_proto_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.proto_class_association[j, j // num_proto_per_class] = 1

        self.backbone = backbone

        if backbone_name == "vgg19":
            self.proj = nn.Sequential(nn.Conv2d(in_channels=512,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=self.dim,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.Sigmoid())
        elif backbone_name == "resnet50":
            self.proj = nn.Sequential(nn.Conv2d(in_channels=2048,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=self.dim,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.Sigmoid())
        else:
            self.proj = nn.Sequential(nn.Conv2d(in_channels=1024,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=self.dim,
                                                out_channels=self.dim,
                                                kernel_size=1),
                                      nn.Sigmoid())

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        self.register_buffer("ones", torch.ones(self.prototype_shape))
        self.fc = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

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
        p2_reshape = p2.view(-1, 1, 1) # shape [num_prototypes, 1, 1]

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # p2_reshape broadcasted to [b, num_prototypes, h, wv]

        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # shape: [b, num_prototypes, h, w]

    def distance_to_similarity(self, distances):
        if self.activation_fn == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        else:
            return -distances

    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        dists = self._l2_dists(f)  # shape: [b, num_prototypes, h, w]
        b, num_prototypes, h, w = dists.shape
        # 2D min pooling
        min_dists = -F.max_pool2d(-dists, kernel_size=(h, w,))  # shape: [b, num_prototypes, 1, 1]
        min_dists = min_dists.squeeze((-1, -2,))  # shape: [b, num_prototypes]

        activations = self.distance_to_similarity(min_dists)  # shape: [b, num_prototypes]
        logits = self.fc(activations)
        return logits, min_dists

    @torch.inference_mode()
    def inference(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        dists = self._l2_dists(f)  # shape: [b, num_prototypes, h, w]
        b, num_prototypes, h, w = dists.shape
        # 2D min pooling
        min_dists = -F.max_pool2d(-dists, kernel_size=(h, w,))  # shape: [b, num_prototypes, 1, 1]
        min_dists = min_dists.squeeze((-1, -2,))  # shape: [b, num_prototypes]

        activations = self.distance_to_similarity(min_dists)  # shape: [b, num_prototypes]
        logits = self.fc(activations)
        return logits, min_dists, dists

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

        self.fc.weight.data.copy_(1 * pos_connections - 0.5 * neg_connections)


class ProtoPNetLoss(nn.Module):
    def __init__(self, l_clst_coef: float, l_sep_coef: float, l_l1_coef: float) -> None:
        super().__init__()
        self.l_clst_coef = l_clst_coef
        self.l_sep_coef = l_sep_coef
        self.l_l1_coef = l_l1_coef
        self.xe = nn.CrossEntropyLoss()

    def forward(self, outputs: tuple[torch.Tensor, torch.Tensor],
                batch: dict[str, torch.Tensor],
                proto_class_association: torch.Tensor,
                fc_weights: torch.Tensor):
        logits, min_dists = outputs
        _, labels, _ = batch
        loss_dict = dict()
        loss_dict["l_y"] = self.xe(logits, labels)

        if self.l_clst_coef != 0 and self.l_sep_coef != 0:
            l_clst, l_sep = self.compute_costs(min_dists,
                                               labels,
                                               proto_class_association)
            loss_dict["l_clst"] = self.l_clst_coef * l_clst
            loss_dict["l_sep"] = self.l_sep_coef * l_sep

        l1_mask = 1 - proto_class_association.T
        l1 = (fc_weights * l1_mask).norm(p=1)
        loss_dict["l_l1"] = self.l_l1_coef * l1
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
