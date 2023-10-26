import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
    return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, T=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.T = T

        self.train_info = {}

    def forward(self, y_s, y_t, target=None):
        assert y_s.ndim == 4

        num_classes = y_s.shape[1]

        y_s = y_s.permute(0, 2, 3, 1).reshape(-1, num_classes)
        y_t = y_t.permute(0, 2, 3, 1).reshape(-1, num_classes)
        p_s = F.softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        inter_loss = inter_class_relation(p_s, p_t)
        intra_loss = intra_class_relation(p_s, p_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss

        self.train_info=dict(
            loss_inter=inter_loss.detach(),
            loss_intra=intra_loss.detach(),
        )

        return loss
