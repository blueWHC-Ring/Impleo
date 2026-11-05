from __future__ import annotations

from typing import Dict

import torch


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    diff2 = (pred - target) ** 2
    denom = mask.sum().clamp_min(1.0)
    return (diff2 * mask).sum() / denom


def huber_elementwise(
    error: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    delta_t = torch.as_tensor(float(delta), dtype=error.dtype, device=error.device)
    abs_error = error.abs()
    return torch.where(
        abs_error <= delta_t,
        0.5 * error * error,
        delta_t * (abs_error - 0.5 * delta_t),
    )


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    loss = huber_elementwise(pred - target, delta=delta)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def masked_pearson_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    min_genes: int = 2,
) -> torch.Tensor:
    """
    Per-cell masked Pearson correlation loss.
    Uses target genes selected by the loss mask.
    Rows require sufficient target genes and nonzero expression variance.
    """
    mask = mask.to(dtype=pred.dtype)
    n = mask.sum(dim=1)
    valid_n = n >= int(min_genes)

    if not torch.any(valid_n):
        return pred.sum() * 0.0

    denom = n.clamp_min(1.0).unsqueeze(1)

    pred_mean = (pred * mask).sum(dim=1, keepdim=True) / denom
    target_mean = (target * mask).sum(dim=1, keepdim=True) / denom

    x = (pred - pred_mean) * mask
    y = (target - target_mean) * mask

    xy = (x * y).sum(dim=1)
    xx = (x * x).sum(dim=1)
    yy = (y * y).sum(dim=1)

    valid = valid_n & (xx > eps) & (yy > eps)

    if not torch.any(valid):
        return pred.sum() * 0.0

    r = xy[valid] / torch.sqrt((xx[valid] * yy[valid]).clamp_min(eps))
    r = torch.clamp(r, min=-1.0, max=1.0)

    return (1.0 - r).mean()


def compute_panel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    corr_lambda: float = 0.05,
    corr_min_genes: int = 2,
) -> Dict[str, torch.Tensor]:
    loss_type = str(loss_type).lower()

    mse = masked_mse(pred, target, mask)
    huber = masked_huber(pred, target, mask, delta=huber_delta)

    zero = pred.sum() * 0.0

    if loss_type == "mse":
        objective = mse
        corr = zero
    elif loss_type == "huber":
        objective = huber
        corr = zero
    elif loss_type in {"huber_corr", "huber_pearson", "huber+pearson"}:
        corr = masked_pearson_corr_loss(
            pred,
            target,
            mask,
            min_genes=corr_min_genes,
        )
        objective = huber + float(corr_lambda) * corr
    else:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}; expected mse, huber, or huber_corr"
        )

    return {
        "loss": objective,
        "masked_mse": mse,
        "masked_huber": huber,
        "corr_loss": corr,
    }
