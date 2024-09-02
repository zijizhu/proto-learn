import torch
from einops import rearrange
from math import sqrt
from models.dino import ProtoDINO
from models.backbone import DINOv2BackboneExpanded, MaskCLIP
from eval_extracted.eval_consistency import evaluate_consistency
from utils.config import load_config_and_logging

def push_forward(self: ProtoDINO, x: torch.Tensor):
    prototype_logits = self.__call__(x, labels=None)["patch_prototype_logits"]
    batch_size, n_patches, C, K = prototype_logits.shape
    H = W = int(sqrt(n_patches))
    return None, rearrange(prototype_logits, "B (H W) C K -> B (C K) H W", H=H, W=W)

if __name__ == "__main__":
    cfg, log_dir, args = load_config_and_logging("logs", return_args=True)

    if "dino" in cfg.model.name:
        backbone = DINOv2BackboneExpanded(name=cfg.model.name, n_splits=cfg.model.n_splits)
        dim = backbone.dim
    elif cfg.model.name.lower().startswith("clip"):
        backbone = MaskCLIP(name=cfg.model.name.split("-", 1)[1])
        dim = 512
    else:
        raise NotImplementedError("Backbone must be one of dinov2 or clip.")
    net = ProtoDINO(
        backbone=backbone,
        dim=dim,
        scale_init=cfg.model.scale_init,
        learn_scale=cfg.model.learn_scale,
        pooling_method=cfg.model.pooling_method,
        cls_head=cfg.model.cls_head,
        pca_compare=cfg.model.pca_compare
    )

    # Monkey-patch the model class to make it compatible with eval script
    ProtoDINO.push_forward = push_forward
    net.img_size = 224
    net.num_prototypes_per_class = 5

    state_dict = torch.load(log_dir / "dino_v2_proto.pth", map_location="cpu")
    net.load_state_dict(state_dict=state_dict)
    net.eval()

    args.data_path = "datasets"
    args.test_batch_size = 64
    args.nb_classes = 200

    score = evaluate_consistency(net, args)
    print(score)
    