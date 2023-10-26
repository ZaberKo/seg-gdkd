import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div, clamp_probs, get_entropy
from .dist import intra_class_relation


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
    def __init__(self, w0: float = 1.0, w1: float = 1.0, w2: float = 2.0, k: int = 3,
                 T: float = 1.0, mask_magnitude=1000, entropy_weight: float = None,
                 ignore_index: int = None, ignore_incorrect: bool = False, resize_out: bool = False):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.k = k
        self.T = T
        self.mask_magnitude = mask_magnitude
        self.entropy_weight = entropy_weight

        self.ignore_index: int = ignore_index
        self.ignore_incorrect: bool = ignore_incorrect
        self.resize_out = resize_out

        self.dist_intra_weight = None

        self.train_info = {}

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes, h, w = y_s.shape[1:]
        H, W = target.shape[1:]

        if self.resize_out:
            # resize model out
            y_s = F.interpolate(
                y_s, (H, W), mode='bilinear', align_corners=True)
            y_t = F.interpolate(
                y_t, (H, W), mode='bilinear', align_corners=True)
        else:
            # resize target
            if self.ignore_incorrect or self.ignore_index:
                target = F.interpolate(
                    target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        target = target.long().flatten()  # [B*h*w]

        loss = self.forward_flatten(y_s, y_t, target)

        return loss

    def forward_flatten(self, y_s, y_t, target):
        self.train_info = {}

        if self.ignore_incorrect:
            pred_t = y_t.argmax(dim=1)
            valid_mask = (target == pred_t)
        elif self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        num_valid = valid_mask.sum()
        num_ignore = target.shape[0]-num_valid.detach()
        self.train_info.update(dict(num_ignore=num_valid))

        if num_ignore == 0:
            loss, p_s, p_t, train_info = self._gdkd_loss(y_s, y_t)
        elif num_ignore <= 1000:
            loss, p_s, p_t, train_info = self._gdkd_loss_mask(
                y_s, y_t, valid_mask)
        else:
            y_s = y_s[valid_mask]
            y_t = y_t[valid_mask]
            target = target[valid_mask]
            loss, p_s, p_t, train_info = self._gdkd_loss(y_s, y_t)

        self.train_info.update(train_info)

        if self.dist_intra_weight is not None:
            dist_intra_loss = intra_class_relation(p_s, p_t)

            loss = loss + self.dist_intra_weight * dist_intra_loss

            self.train_info.update(
                dict(loss_dist_intra=dist_intra_loss.detach())
            )

        return loss

    def _gdkd_loss(self, y_s, y_t):
        mask_u1, mask_u2 = get_masks(y_t, self.k, strategy='best')

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)

        p0_s = cat_mask(p_s, mask_u1, mask_u2)
        p0_t = cat_mask(p_t, mask_u1, mask_u2)

        eps = torch.finfo(y_s.dtype).eps
        num_inf = ((p0_s[:, 1] < eps) | (p0_t[:, 1] < eps)).sum()

        p0_s = clamp_probs(p0_s)
        p0_t = clamp_probs(p0_t)

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
        low_top_loss = kl_div(
            log_p1_s, log_p1_t, self.T)

        # other classes loss
        log_p2_s = F.log_softmax(
            soft_y_s - self.mask_magnitude * mask_u1, dim=1
        )
        log_p2_t = F.log_softmax(
            soft_y_t - self.mask_magnitude * mask_u1, dim=1
        )
        low_other_loss = kl_div(
            log_p2_s, log_p2_t, self.T)

        _soft_y_s = soft_y_s.detach()
        _soft_y_t = soft_y_t.detach()
        train_info = dict(
            loss_high=high_loss.detach(),
            loss_low_top=low_top_loss.detach(),
            loss_low_other=low_other_loss.detach(),
            num_inf=num_inf.detach(),
            soft_y_s_max=_soft_y_s.max(),
            soft_y_s_max_mean=_soft_y_s.max(1)[0].mean(),
            soft_y_s_min=_soft_y_s.min(),
            soft_y_s_min_mean=_soft_y_s.min(1)[0].mean(),
            soft_y_t_max=_soft_y_t.max(),
            soft_y_t_max_mean=_soft_y_t.max(1)[0].mean(),
            soft_y_t_min=_soft_y_t.min(),
            soft_y_t_min_mean=_soft_y_t.min(1)[0].mean(),
        )

        gdkd_loss = (self.w0 * high_loss +
                     self.w1 * low_top_loss +
                     self.w2 * low_other_loss)

        return gdkd_loss, p_s, p_t, train_info

    def _gdkd_loss_mask(self, y_s, y_t, valid_mask):
        train_info = {}
        num_valid = valid_mask.sum()

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)

        if num_valid > 0:
            mask_u1, mask_u2 = get_masks(y_t, self.k, strategy='best')

            p0_s = cat_mask(p_s, mask_u1, mask_u2)
            p0_t = cat_mask(p_t, mask_u1, mask_u2)

            eps = torch.finfo(y_s.dtype).eps
            num_inf = ((p0_s[:, 1] < eps) | (p0_t[:, 1] < eps)).sum()

            p0_s = clamp_probs(p0_s)
            p0_t = clamp_probs(p0_t)

            log_p0_s = torch.log(p0_s)
            high_loss = (
                (F.kl_div(log_p0_s, p0_t, reduction="none")
                 * valid_mask.unsqueeze(1)).sum()
                / num_valid
                * (self.T**2)
            )

            # topk loss
            log_p1_s = F.log_softmax(
                soft_y_s - self.mask_magnitude * mask_u2, dim=1
            )
            log_p1_t = F.log_softmax(
                soft_y_t - self.mask_magnitude * mask_u2, dim=1
            )
            low_top_loss = (
                (F.kl_div(log_p1_s, log_p1_t, reduction="none", log_target=True)
                 * valid_mask.unsqueeze(1)).sum()
                / num_valid
                * (self.T**2)
            )

            # other classes loss
            log_p2_s = F.log_softmax(
                soft_y_s - self.mask_magnitude * mask_u1, dim=1
            )
            log_p2_t = F.log_softmax(
                soft_y_t - self.mask_magnitude * mask_u1, dim=1
            )
            low_other_loss = (
                (F.kl_div(log_p2_s, log_p2_t, reduction="none", log_target=True)
                 * valid_mask.unsqueeze(1)).sum()
                / num_valid
                * (self.T**2)
            )

            _soft_y_s = soft_y_s.detach()
            _soft_y_t = soft_y_t.detach()
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

            num_inf = y_s.new_zeros(())
            logits_info = {}

        gdkd_loss = (self.w0 * high_loss +
                     self.w1 * low_top_loss +
                     self.w2 * low_other_loss)

        train_info.update(dict(
            loss_high=high_loss.detach(),
            loss_low_top=low_top_loss.detach(),
            loss_low_other=low_other_loss.detach(),
            num_inf=num_inf.detach(),
            ** logits_info
        ))

        if num_valid > 0 and self.entropy_weight is not None:
            # add entropy reg term (maximize) to avoid extreme predictions
            entropy = get_entropy(p_s)
            gdkd_loss = gdkd_loss - self.entropy_weight * entropy
            train_info.update(dict(entropy=entropy.detach()))

        return gdkd_loss, p_s, p_t, train_info


class GDKDDIST(GDKD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight
