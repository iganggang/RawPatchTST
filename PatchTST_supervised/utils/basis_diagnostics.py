import math
from typing import Any

import torch


def _resolve_attr_path(obj: Any, attr_path: str) -> Any:
    current = obj
    for part in attr_path.split('.'):
        current = getattr(current, part)
    return current


def resolve_basis_matrix(source: Any, *, attr_path: str = 'B', ckpt_key: str = 'B') -> torch.Tensor:
    """Resolve a basis tensor from tensor/module/checkpoint-like input.

    Supported inputs:
    - torch.Tensor: returned directly
    - nn.Module/object: fetched via dot-path attribute (default: ``B``)
    - dict checkpoint: tries ``ckpt_key`` then ``state_dict[ckpt_key]``
    """

    if source is None:
        raise ValueError('source is None, cannot resolve basis matrix.')

    if isinstance(source, torch.Tensor):
        return source

    if isinstance(source, dict):
        if ckpt_key in source:
            return source[ckpt_key]
        state_dict = source.get('state_dict')
        if isinstance(state_dict, dict) and ckpt_key in state_dict:
            return state_dict[ckpt_key]
        raise KeyError(f'Cannot find key "{ckpt_key}" in checkpoint payload.')

    try:
        return _resolve_attr_path(source, attr_path)
    except AttributeError as exc:
        raise AttributeError(f'Cannot resolve attr_path "{attr_path}" from source {type(source)}.') from exc


def evaluate_basis_change(
    B0: torch.Tensor,
    BT: torch.Tensor,
    *,
    relative_eps: float = 1e-12,
    quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict[str, float | str | bool]:
    """Evaluate how a learnable basis changed from B0 to BT.

    - Auto-casts to float32
    - Auto-moves tensors to the same device
    - Prints concise diagnostics and returns them as a dict
    """

    if B0.ndim != 2 or BT.ndim != 2:
        raise ValueError(f'B0 and BT must be 2D matrices, got shapes {tuple(B0.shape)} and {tuple(BT.shape)}.')
    if B0.shape != BT.shape:
        raise ValueError(f'B0 and BT must share shape, got {tuple(B0.shape)} vs {tuple(BT.shape)}.')

    device = BT.device
    B0f = B0.detach().to(device=device, dtype=torch.float32)
    BTf = BT.detach().to(device=device, dtype=torch.float32)

    delta = BTf - B0f
    delta_fro = torch.linalg.norm(delta, ord='fro')
    b0_fro = torch.linalg.norm(B0f, ord='fro')
    relative = delta_fro / (b0_fro + relative_eps)

    abs_delta = delta.abs().reshape(-1)
    max_abs = abs_delta.max()

    q_tensor = torch.tensor(quantiles, device=device, dtype=torch.float32)
    q_vals = torch.quantile(abs_delta, q_tensor)

    L = B0f.size(0)
    eye = torch.eye(L, device=device, dtype=torch.float32)
    ortho_b0 = torch.linalg.norm(B0f @ B0f.t() - eye, ord='fro')
    ortho_bt = torch.linalg.norm(BTf @ BTf.t() - eye, ord='fro')
    ortho_improved = ortho_bt < ortho_b0

    rel_val = float(relative.item())
    if rel_val < 1e-3:
        interpretation = 'relative < 1e-3: 几乎没学到明显变化'
    elif rel_val < 1e-2:
        interpretation = '1e-3 ~ 1e-2: 轻微变化'
    else:
        interpretation = 'relative > 1e-2: 明显变化'

    result: dict[str, float | str | bool] = {
        'delta_fro': float(delta_fro.item()),
        'relative_delta_fro': rel_val,
        'max_abs_delta': float(max_abs.item()),
        'orthogonality_b0_fro': float(ortho_b0.item()),
        'orthogonality_bt_fro': float(ortho_bt.item()),
        'orthogonality_improved': bool(ortho_improved.item() if isinstance(ortho_improved, torch.Tensor) else ortho_improved),
        'interpretation': interpretation,
    }

    for q, v in zip(quantiles, q_vals):
        result[f'abs_delta_q{int(math.floor(q * 100))}'] = float(v.item())

    print('[basis-change] ||BT-B0||_F = {:.6e}'.format(result['delta_fro']))
    print('[basis-change] relative = {:.6e}'.format(result['relative_delta_fro']))
    print('[basis-change] max|BT-B0| = {:.6e}'.format(result['max_abs_delta']))
    quantile_str = ', '.join(
        [f'q{int(math.floor(q * 100))}={result[f"abs_delta_q{int(math.floor(q * 100))}"]:.6e}' for q in quantiles]
    )
    print(f'[basis-change] |BT-B0| quantiles: {quantile_str}')
    print('[basis-change] ||B0B0^T-I||_F = {:.6e}, ||BTBT^T-I||_F = {:.6e}, improved={}'.format(
        result['orthogonality_b0_fro'],
        result['orthogonality_bt_fro'],
        result['orthogonality_improved'],
    ))
    print(f'[basis-change] interpretation: {interpretation}')

    return result
