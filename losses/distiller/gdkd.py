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


def _gdkd_loss_fn(y_s, y_t, w0, w1, w2, k, T, mask_magnitude, kl_type):
    mask_u1, mask_u2 = get_masks(y_t, k, strategy='best')

    soft_y_s = y_s / T
    soft_y_t = y_t / T

    p_s = F.softmax(soft_y_s, dim=1)
    p_t = F.softmax(soft_y_t, dim=1)
    p0_s = cat_mask(p_s, mask_u1, mask_u2)
    p0_t = cat_mask(p_t, mask_u1, mask_u2)

    log_p0_s = torch.log(p0_s)
    high_loss = (
        F.kl_div(log_p0_s, p0_t, reduction="batchmean")
        * (T**2)
    )

    # topk loss
    log_p1_s = F.log_softmax(
        soft_y_s - mask_magnitude * mask_u2, dim=1
    )
    log_p1_t = F.log_softmax(
        soft_y_t - mask_magnitude * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_s, log_p1_t, T, kl_type=kl_type)

    # other classes loss
    log_p2_s = F.log_softmax(
        soft_y_s - mask_magnitude * mask_u1, dim=1
    )
    log_p2_t = F.log_softmax(
        soft_y_t - mask_magnitude * mask_u1, dim=1
    )

    low_other_loss = kl_div(
        log_p2_s, log_p2_t, T, kl_type=kl_type)

    gdkd_loss = (w0 * high_loss +
                 w1 * low_top_loss +
                 w2 * low_other_loss)

    _soft_y_s = soft_y_s.detach()
    _soft_y_t = soft_y_t.detach()

    train_info = dict(
        loss_high=high_loss.detach(),
        loss_low_top=low_top_loss.detach(),
        loss_low_other=low_other_loss.detach(),
        soft_y_s_max=_soft_y_s.max(),
        soft_y_s_max_mean=_soft_y_s.max(1)[0].mean(),
        soft_y_s_min=_soft_y_s.min(),
        soft_y_s_min_mean=_soft_y_s.min(1)[0].mean(),
        soft_y_t_max=_soft_y_t.max(),
        soft_y_t_max_mean=_soft_y_t.max(1)[0].mean(),
        soft_y_t_min=_soft_y_t.min(),
        soft_y_t_min_mean=_soft_y_t.min(1)[0].mean(),
    )

    return gdkd_loss, train_info, p_s, p_t


def _gdkd_loss_fn2(y_s, y_t, valid_mask, w0, w1, w2, k, T, mask_magnitude, kl_type):
    num_valid = valid_mask.sum()

    if num_valid > 0:
        mask_u1, mask_u2 = get_masks(y_t, k, strategy='best')

        soft_y_s = y_s / T
        soft_y_t = y_t / T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)
        p0_s = cat_mask(p_s, mask_u1, mask_u2)
        p0_t = cat_mask(p_t, mask_u1, mask_u2)

        log_p0_s = torch.log(p0_s)
        high_loss = (
            (F.kl_div(log_p0_s, p0_t, reduction="none")
             * valid_mask.unsqueeze(1)).sum()
            / num_valid
            * (T**2)
        )

        # topk loss
        log_p1_s = F.log_softmax(
            soft_y_s - mask_magnitude * mask_u2, dim=1
        )
        log_p1_t = F.log_softmax(
            soft_y_t - mask_magnitude * mask_u2, dim=1
        )

        low_top_loss = (
            kl_div(log_p1_s, log_p1_t, T, kl_type=kl_type, reduction="none")
            * valid_mask
        ).sum() / num_valid

        # other classes loss
        log_p2_s = F.log_softmax(
            soft_y_s - mask_magnitude * mask_u1, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - mask_magnitude * mask_u1, dim=1
        )

        low_other_loss = (
            kl_div(log_p2_s, log_p2_t, T, kl_type=kl_type, reduction="none")
            * valid_mask
        ).sum() / num_valid

        _soft_y_s = soft_y_s.detach()[valid_mask]
        _soft_y_t = soft_y_t.detach()[valid_mask]

        logits_info = dict(
            soft_y_s_max=_soft_y_s.max(),
            soft_y_s_max_mean=_soft_y_s.max(1)[0].mean(),
            soft_y_s_min=_soft_y_s.min(),
            soft_y_s_min_mean=_soft_y_s.min(1)[0].mean(),
            soft_y_t_max=_soft_y_t.max(),
            soft_y_t_max_mean=_soft_y_t.max(1)[0].mean(),
            soft_y_t_min=_soft_y_t.min(),
            soft_y_t_min_mean=_soft_y_t.min(1)[0].mean(),
        )
    else:
        high_loss = 0.0 * y_s.sum()
        low_top_loss = 0.0 * y_s.sum()
        low_other_loss = 0.0 * y_s.sum()

        logits_info = {}

    gdkd_loss = (w0 * high_loss +
                 w1 * low_top_loss +
                 w2 * low_other_loss)

    train_info = dict(
        loss_high=high_loss.detach(),
        loss_low_top=low_top_loss.detach(),
        loss_low_other=low_other_loss.detach(),
        num_valid=num_valid.detach(),
        **logits_info
    )

    return gdkd_loss, train_info, p_s, p_t


class GDKD(nn.Module):
    def __init__(self, w0: float = 1.0, w1: float = 1.0, w2: float = 2.0, k: int = 3, T: float = 1.0, mask_magnitude=1000, kl_type="forward", ignore_index=None):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.k = k
        self.T = T
        self.mask_magnitude = mask_magnitude
        self.kl_type = kl_type

        self.ignore_index = ignore_index

        self.dist_intra_weight = None

        self.train_info = {}

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes, h, w = y_s.shape[1:]

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        if self.ignore_index:
            target = F.interpolate(
                target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
            target = target.long().flatten()  # [B*h*w]
            valid_mask = (target != self.ignore_index)
            loss, self.train_info, p_s, p_t = _gdkd_loss_fn2(
                y_s=y_s,
                y_t=y_t,
                valid_mask=valid_mask,
                w0=self.w0,
                w1=self.w1,
                w2=self.w2,
                k=self.k,
                T=self.T,
                mask_magnitude=self.mask_magnitude,
                kl_type=self.kl_type,
            )
        else:
            loss, self.train_info, p_s, p_t = _gdkd_loss_fn(
                y_s=y_s,
                y_t=y_t,
                w0=self.w0,
                w1=self.w1,
                w2=self.w2,
                k=self.k,
                T=self.T,
                mask_magnitude=self.mask_magnitude,
                kl_type=self.kl_type,
            )

        if self.dist_intra_weight is not None:
            dist_intra_loss = intra_class_relation(p_s, p_t)

            loss = loss + self.dist_intra_weight * dist_intra_loss

            self.train_info.update(
                dict(loss_dist_intra=dist_intra_loss.detach())
            )

        return loss


class GDKDDIST(GDKD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight
