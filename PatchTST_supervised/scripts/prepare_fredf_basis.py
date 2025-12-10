import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from data_provider.data_factory import data_dict


def collect_train_windows(args: argparse.Namespace) -> torch.Tensor:
    """Collect all training label windows for PCA basis learning."""
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    dataset = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    label_windows = []
    f_dim = -1 if args.features == 'MS' else 0
    for _, batch_y, _, _ in loader:
        label_windows.append(batch_y[:, -args.pred_len :, f_dim:])

    return torch.cat(label_windows, dim=0)


def learn_pca_basis(y_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Learn PCA basis following the FreDF-PCA recipe."""
    if y_all.ndim != 3:
        raise ValueError(f"Expected [N, T, C] label tensor, got shape {tuple(y_all.shape)}")

    _, t, _ = y_all.shape
    y_flat = y_all.reshape(-1, t)
    mean = y_flat.mean(dim=0, keepdim=True)
    y_centered = y_flat - mean

    cov = (y_centered.T @ y_centered) / y_centered.size(0)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]
    basis = eigvecs.T
    return basis, mean


def main():
    parser = argparse.ArgumentParser(description='Prepare FreDF-PCA basis for LFT loss')
    parser.add_argument('--data', type=str, required=True, help='dataset name (e.g., ETTh1, ETTm1)')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task type')
    parser.add_argument('--target', type=str, default='OT', help='target feature for S/MS settings')
    parser.add_argument('--freq', type=str, default='h', help='frequency string for timestamp encoding')
    parser.add_argument('--embed', type=str, default='timeF', help='embedding type, matches training argument')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='label sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used for iterating the train set')
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--output_path', type=str, required=True, help='where to store the learned PCA basis (torch file)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print('Collecting training label windows...')
    y_all = collect_train_windows(args)
    print(f'Collected {y_all.shape[0]} windows of shape [{args.pred_len}, {y_all.shape[-1]}].')

    print('Learning FreDF-PCA basis...')
    basis, mean = learn_pca_basis(y_all)

    torch.save({'basis': basis, 'mean': mean}, args.output_path)
    print(f'Basis saved to {args.output_path}')


if __name__ == '__main__':
    main()
