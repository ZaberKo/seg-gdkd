import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div
from .dist_kd import intra_class_relation


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
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, T: float = 1.0, mask_magnitude=1000, kl_type="forward"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.mask_magnitude = mask_magnitude
        self.kl_type = kl_type

        self.train_info = {}

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes = y_s.shape[1]

        h, w = y_s.shape[-2:]
        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        
        # resize target to pred size
        # TODO: fixit
        target = F.interpolate(
            target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
        target = target.long().flatten()

        gt_mask = _get_gt_mask(y_t, target)
        other_mask = _get_other_mask(y_t, target)

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)
        p0_s = cat_mask(p_s, gt_mask, other_mask)
        p0_t = cat_mask(p_t, gt_mask, other_mask)

        log_p0_s = torch.log(p0_s)
        tckd_loss = (
            F.kl_div(log_p0_s, p0_t, reduction="batchmean")
            * (self.T**2)
        )

        log_p2_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * gt_mask, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * gt_mask, dim=1
        )

        nckd_loss = kl_div(log_p2_s,
                           log_p2_t, self.T, kl_type=self.kl_type)

        dkd_loss = self.alpha * tckd_loss + self.beta * nckd_loss

        self.train_info = dict(
            loss_tckd=tckd_loss.detach(),
            loss_nckd=nckd_loss.detach()
        )

        return dkd_loss


class DKDDIST(DKD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes = y_s.shape[1]

        h, w = y_s.shape[-2:]
        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        
        # resize target to pred siz
        target = F.interpolate(
            target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
        target = target.long().flatten()

        gt_mask = _get_gt_mask(y_t, target)
        other_mask = _get_other_mask(y_t, target)

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)

        p0_s = cat_mask(p_s, gt_mask, other_mask)
        p0_t = cat_mask(p_t, gt_mask, other_mask)

        log_p0_s = torch.log(p0_s)
        tckd_loss = (
            F.kl_div(log_p0_s, p0_t, reduction="batchmean")
            * (self.T**2)
        )

        log_p2_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * gt_mask, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * gt_mask, dim=1
        )

        nckd_loss = kl_div(log_p2_s,
                           log_p2_t, self.T, kl_type=self.kl_type)

        dkd_loss = self.alpha * tckd_loss + self.beta * nckd_loss

        dist_intra_loss = intra_class_relation(p_s, p_t)

        loss = dkd_loss + self.dist_intra_weight * dist_intra_loss

        self.train_info = dict(
            loss_tckd=tckd_loss.detach(),
            loss_nckd=nckd_loss.detach(),
            loss_dist_intra=dist_intra_loss.detach()
        )

        return loss
