import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div, clamp_probs, get_entropy
from .dist import intra_class_relation


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, T: float = 1.0, mask_magnitude=1000):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.mask_magnitude = mask_magnitude

        self.dist_intra_weight = None
        self.ignore_index = -1

        self.train_info = {}

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes, h, w = y_s.shape[1:]
        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        # resize target to pred size (use nearest?)
        target = F.interpolate(
            target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
        target = target.long().flatten()  # [B*h*w]

        # move target -1 to 0, then use mask
        valid_mask = target != self.ignore_index  # [B*h*w]
        target[~valid_mask] = 0
        num_valid = valid_mask.sum()  # keep as tensor
        # accum_batch_size = ingore_mask.shape[0]

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)

        if num_valid > 0:
            gt_mask = _get_gt_mask(y_t, target)
            other_mask = _get_other_mask(y_t, target)

            p0_s = cat_mask(p_s, gt_mask, other_mask)
            p0_t = cat_mask(p_t, gt_mask, other_mask)

            eps = torch.finfo(y_s.dtype).eps
            num_inf = ((p0_s[:, 1] < eps) | (p0_t[:, 1] < eps)).sum()

            p0_s = clamp_probs(p0_s)
            p0_t = clamp_probs(p0_t)

            log_p0_s = torch.log(p0_s)
            tckd_loss = (
                (F.kl_div(log_p0_s, p0_t, reduction="none")
                 * valid_mask.unsqueeze(1)).sum()
                / num_valid
                * (self.T**2)
            )

            log_p2_s = F.log_softmax(
                soft_y_s - self.mask_magnitude * gt_mask, dim=1
            )
            log_p2_t = F.log_softmax(
                soft_y_t - self.mask_magnitude * gt_mask, dim=1
            )

            nckd_loss = (
                (F.kl_div(log_p2_s, log_p2_t, reduction="none", log_target=True)
                 * valid_mask.unsqueeze(1)).sum()
                / num_valid
                * (self.T**2)
            )
        else:
            tckd_loss = 0.0 * y_s.sum()
            nckd_loss = 0.0 * y_s.sum()

            num_inf = y_s.new_zeros(())

        self.train_info = dict(
            loss_tckd=tckd_loss.detach(),
            loss_nckd=nckd_loss.detach(),
            num_valid=num_valid.detach(),
            num_inf=num_inf.detach(),
            entropy=get_entropy(p_s.detach())
        )

        loss = self.alpha * tckd_loss + self.beta * nckd_loss

        if self.dist_intra_weight is not None:
            dist_intra_loss = intra_class_relation(p_s, p_t)

            loss = loss + self.dist_intra_weight * dist_intra_loss

            self.train_info.update(
                dict(loss_dist_intra=dist_intra_loss.detach())
            )

        return loss


class DKDDIST(DKD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight
