import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import kl_div
from .dist_kd import intra_class_relation


class KD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, T: float = 1.0, kl_type="forward", ignore_index=None):
        super().__init__()
        self.T = T
        self.kl_type = kl_type

        self.ignore_index = ignore_index

        self.dist_intra_weight = None

        self.train_info = {}

    def forward(self, y_s, y_t, target=None):
        assert y_s.ndim == 4

        num_classes, h, w = y_s.shape[1:]

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T

        log_p_s = F.log_softmax(soft_y_s, dim=1)
        log_p_t = F.log_softmax(soft_y_t, dim=1)

        if self.ignore_index:
            target = F.interpolate(
                target.float().unsqueeze(1), (h, w), mode='nearest').squeeze(1)
            target = target.long().flatten()  # [B*h*w]
            valid_mask = (target != self.ignore_index)
            num_valid = valid_mask.sum()
            # loss = kl_div(log_p_s, log_p_t, self.T, valid_mask, kl_type=self.kl_type)
            loss = (
                kl_div(log_p_s, log_p_t, self.T, kl_type=self.kl_type,
                       reduction="none")*valid_mask
            ).sum() / num_valid

        else:
            loss = kl_div(log_p_s, log_p_t, self.T, kl_type=self.kl_type)

        if self.dist_intra_weight is not None:
            p_s = F.softmax(soft_y_s, dim=1)
            p_t = F.softmax(soft_y_t, dim=1)
            dist_intra_loss = intra_class_relation(p_s, p_t)
            loss = loss + self.dist_intra_weight * dist_intra_loss

            self.train_info = dict(
                loss_dist_intra=dist_intra_loss.detach()
            )

        return loss


class KDDIST(KD):
    def __init__(self, dist_intra_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.dist_intra_weight = dist_intra_weight
