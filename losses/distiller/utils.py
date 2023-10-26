import torch
import torch.nn.functional as F



def kl_div(log_p, log_q, T, kl_type='forward', reduction="batchmean"):
    if kl_type == "forward":
        res = F.kl_div(log_p, log_q, reduction=reduction,
                       log_target=True)
    elif kl_type == "reverse":
        res = F.kl_div(log_q, log_p, reduction=reduction,
                       log_target=True)
    elif kl_type == "both":
        res = 0.5 * (
            F.kl_div(log_p, log_q, reduction=reduction, log_target=True) +
            F.kl_div(log_q, log_p, reduction=reduction, log_target=True)
        )
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    if reduction == "none":
        res = res.sum(dim=1)  # [B,C]->[B]

    res = res * (T**2)

    return res


def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)

def get_entropy(probs):
    probs = clamp_probs(probs)
    entropy = - (probs.log() * probs).sum(-1)
    return entropy.mean()