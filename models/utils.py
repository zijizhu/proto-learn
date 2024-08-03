import torch
import torch.nn.functional as F
from abc import ABC
from torch import nn


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


class PPC(nn.Module, ABC):
    def __init__(self):
        super(PPC, self).__init__()

        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)

        return loss_ppc
    
    
class FSCELoss(nn.Module):
    def __init__(self, weight: list[float] | None = None):
        super(FSCELoss, self).__init__()
        # weight = [0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754,
        #           1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037,
        #           1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
        if weight:
            weight = torch.tensor(weight, dtype=torch.float32)

        ignore_index = -1

        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="mean")

    def forward(self, inputs, *targets, weights=None, **kwargs):
        target = targets[0]
        return self.ce_loss(inputs, target)
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            # target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class PPD(nn.Module, ABC):
    def __init__(self):
        super(PPD, self).__init__()

        self.ignore_label = -1

    def forward(self, contrast_logits, contrast_target):
        contrast_logits = contrast_logits[contrast_target != self.ignore_label, :]
        contrast_target = contrast_target[contrast_target != self.ignore_label]

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        loss_ppd = (1 - logits).pow(2).mean()

        return loss_ppd


class PixelPrototypeCELoss(nn.Module, ABC):
    def __init__(self):
        super(PixelPrototypeCELoss, self).__init__()

        self.loss_ppc_weight = 0.01
        self.loss_ppd_weight = 0.001

        self.seg_criterion = FSCELoss()

        self.ppc_criterion = PPC()
        self.ppd_criterion = PPD()

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if isinstance(preds, dict):
            assert "seg" in preds
            assert "logits" in preds
            assert "target" in preds

            seg = preds['seg']
            contrast_logits = preds['logits']
            contrast_target = preds['target']
            loss_ppc = self.ppc_criterion(contrast_logits, contrast_target)
            loss_ppd = self.ppd_criterion(contrast_logits, contrast_target)

            # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            pred = seg
            loss = self.seg_criterion(pred, target)
            return loss + self.loss_ppc_weight * loss_ppc + self.loss_ppd_weight * loss_ppd

        seg = preds
        pred = seg
        # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        return loss