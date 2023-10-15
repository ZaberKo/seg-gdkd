import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div
from .dist_kd import intra_class_relation


def get_masks(logits, k=5, strategy="best"):
    if strategy == "best":
        largest_flag = True
    elif strategy == "worst":
        largest_flag = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = torch.topk(logits, k, dim=-1,
                       largest=largest_flag,
                       sorted=False).indices

    # topk mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


class GDKD(nn.Module):
    def __init__(self, w0: float = 1.0, w1: float = 1.0, w2: float = 2.0, k: int = 3, T: float = 1.0, mask_magnitude=1000, kl_type="forward"):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.k = k
        self.T = T
        self.mask_magnitude = mask_magnitude
        self.kl_type = kl_type

        self.train_info = {}

    def forward(self, y_s, y_t, target=None):
        assert y_s.ndim == 4

        num_classes = y_s.shape[1]

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)

        mask_u1, mask_u2 = get_masks(y_t, self.k, strategy='best')

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)
        p0_s = cat_mask(p_s, mask_u1, mask_u2)
        p0_t = cat_mask(p_t, mask_u1, mask_u2)

        log_p0_s = torch.log(p0_s)
        high_loss = (
            F.kl_div(log_p0_s, p0_t, reduction="batchmean")
            * (self.T**2)
        )

        # topk loss
        log_p1_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * mask_u2, dim=1
        )
        log_p1_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * mask_u2, dim=1
        )

        low_top_loss = kl_div(log_p1_s, log_p1_t, self.T, kl_type=self.kl_type)

        # other classes loss
        log_p2_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * mask_u1, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * mask_u1, dim=1
        )

        low_other_loss = kl_div(
            log_p2_s, log_p2_t, self.T, kl_type=self.kl_type)

        gdkd_loss = (self.w0 * high_loss + self.w1 *
                     low_top_loss + self.w2 * low_other_loss)

        self.train_info = dict(
            loss_high=high_loss.detach(),
            loss_low_top=low_top_loss.detach(),
            loss_low_other=low_other_loss.detach()
        )

        return gdkd_loss


class GDKDDIST(GDKD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight

    def forward(self, y_s, y_t, target=None):
        assert y_s.ndim == 4

        num_classes = y_s.shape[1]

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)

        mask_u1, mask_u2 = get_masks(y_t, self.k, strategy='best')

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)
        p0_s = cat_mask(p_s, mask_u1, mask_u2)
        p0_t = cat_mask(p_t, mask_u1, mask_u2)

        log_p0_s = torch.log(p0_s)
        high_loss = (
            F.kl_div(log_p0_s, p0_t, reduction="batchmean")
            * (self.T**2)
        )

        # topk loss
        log_p1_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * mask_u2, dim=1
        )
        log_p1_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * mask_u2, dim=1
        )

        low_top_loss = kl_div(log_p1_s, log_p1_t, self.T, kl_type=self.kl_type)

        # other classes loss
        log_p2_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * mask_u1, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * mask_u1, dim=1
        )

        low_other_loss = kl_div(
            log_p2_s, log_p2_t, self.T, kl_type=self.kl_type)

        gdkd_loss = (self.w0 * high_loss + self.w1 *
                     low_top_loss + self.w2 * low_other_loss)

        dist_intra_loss = intra_class_relation(p_s, p_t)

        loss = gdkd_loss + self.dist_intra_weight * dist_intra_loss

        self.train_info = dict(
            loss_high=high_loss.detach(),
            loss_low_top=low_top_loss.detach(),
            loss_low_other=low_other_loss.detach(),
            loss_dist_intra=dist_intra_loss.detach()
        )

        return loss
