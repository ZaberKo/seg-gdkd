import torch.nn as nn
import torch.nn.functional as F

from .utils import get_entropy


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    """
        computate pearson correlation between x and y at last dim
        x: Tensor[B, ... , C]
        y: Tensor[B, ... , C]

        return: Tensor[B, ...]
    """
    return cosine_similarity(
        x - x.mean(dim=-1).unsqueeze(-1),
        y - y.mean(dim=-1).unsqueeze(-1),
        eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def per_image_intra_class_relation(y_s, y_t, batch_size, num_classes):
    y_s = y_s.reshape(batch_size, -1, num_classes)  # [B*h*w,C] -> [B, h*w, C]
    y_t = y_t.reshape(batch_size, -1, num_classes)  # [B*h*w,C] -> [B, h*w, C]

    return intra_class_relation(y_s.transpose(1, 2), y_t.transpose(1, 2))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, T=1.0, 
                 resize_out: bool = False, per_image_intra=False):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.T = T

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

        # [B,C,h,w] -> [B,h,w,C] -> [B*h*w,C]
        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)

        loss = self.forward_flatten(y_s, y_t, target=None)

        return loss

    def forward_flatten(self, y_s, y_t, target=None):
        soft_y_s = y_s / self.T
        soft_y_t = y_t / self.T
        p_s = F.softmax(soft_y_s, dim=1)
        p_t = F.softmax(soft_y_t, dim=1)
        inter_loss = inter_class_relation(p_s, p_t)

        if self.per_image_intra:
            intra_loss = per_image_intra_class_relation(
                p_s, p_t, self.meta_info['batch_size'], self.meta_info['num_classes']
            )
        else:
            intra_loss = intra_class_relation(p_s, p_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss

        _soft_y_s = soft_y_s.detach()
        _soft_y_t = soft_y_t.detach()
        self.train_info = dict(
            loss_inter=inter_loss.detach(),
            loss_intra=intra_loss.detach(),
            entropy=get_entropy(p_s.detach()),
            soft_y_s_max=_soft_y_s.max(),
            soft_y_s_max_mean=_soft_y_s.max(1)[0].mean(),
            soft_y_s_min=_soft_y_s.min(),
            soft_y_s_min_mean=_soft_y_s.min(1)[0].mean(),
            soft_y_t_max=_soft_y_t.max(),
            soft_y_t_max_mean=_soft_y_t.max(1)[0].mean(),
            soft_y_t_min=_soft_y_t.min(),
            soft_y_t_min_mean=_soft_y_t.min(1)[0].mean(),
        )

        return loss
