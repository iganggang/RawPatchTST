import torch
import torch.nn as nn


def huber_loss(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_res = residual.abs()
    quadratic = 0.5 * abs_res ** 2
    linear = delta * (abs_res - 0.5 * delta)
    return torch.where(abs_res <= delta, quadratic, linear)


class NoiseAdaptiveHybridHuber(nn.Module):
    """Volatility-adaptive hybrid Huber loss with optional reduction."""

    requires_volatility = True

    def __init__(self, delta: float = 1.0, gamma: float = 1.0, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, x_for_volatility: torch.Tensor | None = None) -> torch.Tensor:
        if x_for_volatility is None:
            x_for_volatility = y_true

        batch_size = x_for_volatility.size(0)
        std_per_sample = x_for_volatility.view(batch_size, -1).std(dim=1)

        ref_scale = std_per_sample.mean().detach() + self.eps
        rel_std = std_per_sample / ref_scale

        weights = 1.0 / (1.0 + self.gamma * rel_std)
        weights = weights.view(batch_size, 1, 1)

        residual = y_pred - y_true
        mse_term = 0.5 * residual ** 2
        huber_term = huber_loss(residual, delta=self.delta)

        loss = weights * mse_term + (1.0 - weights) * huber_term

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()
