import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div, get_entropy
from .dist import intra_class_relation, per_image_intra_class_relation


class KD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, T: float = 1.0,
                 ignore_index=None, resize_out: bool = False,  per_image_intra=False):
        super().__init__()
        self.T = T

        self.ignore_index = ignore_index

        self.dist_intra_weight = None
        self.resize_out = resize_out
        self.per_image_intra = per_image_intra

        self.meta_info = {}
        self.train_info = {}

    def forward(self, y_s, y_t, target):
        assert y_s.ndim == 4

        num_classes, h, w = y_s.shape[1:]
        H, W = target.shape[1:]

        self.meta_info=dict(
            batch_size = y_s.shape[0],
            num_classes = num_classes,
            feature_size = (h, w),
            img_size = (H, W),
        )

        if self.resize_out:
            # resize model out
            y_s = F.interpolate(
                y_s, (H, W), mode='bilinear', align_corners=True)
            y_t = F.interpolate(
                y_t, (H, W), mode='bilinear', align_corners=True)
        else:
            # resize target
            if self.ignore_index:
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

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        log_p_s = F.log_softmax(soft_y_s, dim=1)
        log_p_t = F.log_softmax(soft_y_t, dim=1)

        if self.ignore_index:
            valid_mask = (target != self.ignore_index)
            num_valid = valid_mask.sum()

            if num_valid > 0:
                loss = (
                    (F.kl_div(log_p_s, log_p_t, reduction="none", log_target=True)
                     * valid_mask.unsqueeze(1)).sum()
                    / num_valid
                    * (self.T**2)
                )
            else:
                loss = 0.0 * y_s.sum()

        else:
            loss = kl_div(log_p_s, log_p_t, self.T)

        p_s = None
        if self.dist_intra_weight is not None:
            p_s = F.softmax(soft_y_s, dim=1)
            p_t = F.softmax(soft_y_t, dim=1)
            if self.per_image_intra:
                dist_intra_loss = per_image_intra_class_relation(
                    p_s, p_t, self.meta_info['batch_size'], self.meta_info['num_classes']
                )
            else:
                dist_intra_loss = intra_class_relation(p_s, p_t)
            loss = loss + self.dist_intra_weight * dist_intra_loss

            self.train_info.update(dict(
                loss_dist_intra=dist_intra_loss.detach()
            ))

        if p_s is None:
            p_s = F.softmax(soft_y_s, dim=1)

        _soft_y_s = soft_y_s.detach()
        _soft_y_t = soft_y_t.detach()
        self.train_info.update(dict(
            entropy=get_entropy(p_s.detach()),
            soft_y_s_max=_soft_y_s.max(),
            soft_y_s_max_mean=_soft_y_s.max(1)[0].mean(),
            soft_y_s_min=_soft_y_s.min(),
            soft_y_s_min_mean=_soft_y_s.min(1)[0].mean(),
            soft_y_t_max=_soft_y_t.max(),
            soft_y_t_max_mean=_soft_y_t.max(1)[0].mean(),
            soft_y_t_min=_soft_y_t.min(),
            soft_y_t_min_mean=_soft_y_t.min(1)[0].mean(),
        ))

        return loss


class KDDIST(KD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight
