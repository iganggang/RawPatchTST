import torch
import torch.nn as nn


class LearnableTransform(nn.Module):
    """Trainable linear transform along the temporal dimension."""

    def __init__(self, T: int, init_B: torch.Tensor | None = None):
        super().__init__()
        self.T = T
        if init_B is None:
            B = torch.eye(T)
        else:
            if init_B.shape != (T, T):
                raise ValueError(f'Expected init_B to have shape {(T, T)}, got {tuple(init_B.shape)}.')
            B = init_B.clone()
        self.B = nn.Parameter(B)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Apply the learnable transform to a batch of sequences.

        Args:
            Y: Tensor shaped ``[B, T, C]``.

        Returns:
            Transformed tensor with shape ``[B, T, C]``.
        """

        batch_size, time_steps, channels = Y.shape
        if time_steps != self.T:
            raise ValueError(
                f'Input length {time_steps} mismatches transform length {self.T}.'
            )

        Y_flat = Y.permute(0, 2, 1).reshape(-1, time_steps)
        Z_flat = Y_flat @ self.B.t()
        return Z_flat.reshape(batch_size, channels, time_steps).permute(0, 2, 1)


def orthogonal_regularizer(B: torch.Tensor) -> torch.Tensor:
    """Penalize deviation from orthogonality."""

    time_steps = B.size(0)
    identity = torch.eye(time_steps, device=B.device, dtype=B.dtype)
    return ((B @ B.t() - identity) ** 2).mean()


def project_to_basis(y: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project batch of sequences onto an orthonormal basis.

    Args:
        y: Input tensor shaped ``[B, T, C]``.
        basis: Orthonormal basis with shape ``[T, T]`` whose rows are basis functions.
    """
    bsz, time_steps, channels = y.shape
    if basis.shape[0] != basis.shape[1] or basis.shape[0] != time_steps:
        raise ValueError(
            f"Basis shape {tuple(basis.shape)} is incompatible with series length {time_steps}."
        )

    y_flat = y.permute(0, 2, 1).reshape(-1, time_steps)
    z_flat = y_flat @ basis.t()
    return z_flat.reshape(bsz, channels, time_steps).permute(0, 2, 1)


def huber_loss(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Element-wise Huber loss.

    Args:
        residual: Difference between prediction and target.
        delta: Transition point between quadratic and linear.
    """
    abs_res = residual.abs()
    quadratic = 0.5 * abs_res ** 2
    linear = delta * (abs_res - 0.5 * delta)
    return torch.where(abs_res <= delta, quadratic, linear)


class NoiseAdaptiveHybridHuber(nn.Module):
    """Volatility-adaptive hybrid Huber loss.

    This combines MSE and Huber losses with a data-driven weight derived
    from per-sample volatility. Set ``reduction='none'`` to obtain the
    unreduced loss map.
    """

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


class FreDFPCALoss(nn.Module):
    """Linear Frequency-domain Transformation (LFT) loss using a PCA basis.

    Combines frequency-domain L1 distance computed after projecting onto an
    offline PCA basis with a time-domain MSE term.
    """

    def __init__(self, basis: torch.Tensor, alpha: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
            raise ValueError('PCA basis must be a square matrix of shape [T, T].')
        if not (0.0 <= alpha <= 1.0):
            raise ValueError('alpha must lie in [0, 1].')

        self.register_buffer('basis', basis)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f'Mismatched shapes for LFT loss: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}.'
            )

        z_pred = project_to_basis(y_pred, self.basis)
        z_true = project_to_basis(y_true, self.basis)

        freq_term = torch.abs(z_pred - z_true)
        time_term = (y_pred - y_true) ** 2

        if self.reduction == 'sum':
            freq_loss = freq_term.sum()
            time_loss = time_term.sum()
        elif self.reduction == 'none':
            freq_loss = freq_term
            time_loss = time_term
        else:
            freq_loss = freq_term.mean()
            time_loss = time_term.mean()

        return self.alpha * freq_loss + (1.0 - self.alpha) * time_loss


class FreDFLearnableLoss(nn.Module):
    """End-to-end learnable FreDF loss with orthogonality regularization."""

    requires_volatility = False

    def __init__(
        self,
        T: int,
        alpha: float = 0.8,
        beta: float = 1e-2,
        init_basis: torch.Tensor | None = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError('alpha must lie in [0, 1].')
        if beta < 0:
            raise ValueError('beta must be non-negative.')

        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.transform = LearnableTransform(T, init_B=init_basis)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f'Mismatched shapes for learnable FreDF loss: {tuple(y_pred.shape)} vs {tuple(y_true.shape)}.'
            )

        z_pred = self.transform(y_pred)
        z_true = self.transform(y_true)

        freq_term = torch.abs(z_pred - z_true)
        time_term = (y_pred - y_true) ** 2
        ortho_term = orthogonal_regularizer(self.transform.B)

        if self.reduction == 'sum':
            freq_loss = freq_term.sum()
            time_loss = time_term.sum()
        elif self.reduction == 'none':
            freq_loss = freq_term
            time_loss = time_term
        else:
            freq_loss = freq_term.mean()
            time_loss = time_term.mean()

        return self.alpha * freq_loss + (1.0 - self.alpha) * time_loss + self.beta * ortho_term
