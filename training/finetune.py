"""Fine-tune all Impleo parameters by masked reconstruction of a measured panel."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from impleo.inference import forward_panel, load_model, read_names, validate_panel
from impleo.losses import compute_panel_loss


def make_splits(n, seed=20261024):
    if n < 5:
        raise ValueError('At least five cells are required')
    order = np.random.default_rng(seed).permutation(n)
    nv, nt = max(1, round(n * .1)), max(1, round(n * .2))
    return dict(train=order[nt + nv:], val=order[nt:nt + nv], test=order[:nt])


def check_splits(splits, n):
    parts = {k: np.asarray(splits[k]) for k in ('train', 'val', 'test')}
    if any(x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) or not len(x) for x in parts.values()):
        raise ValueError('Each split must be a nonempty integer row-index vector')
    joined = np.concatenate(list(parts.values()))
    if not np.array_equal(np.sort(joined), np.arange(n)):
        raise ValueError('Splits must partition all cells exactly once')
    return parts


def finetune(checkpoint, values, panel, output_dir, splits=None, seed=42,
             epochs=5, device='cuda', modality='seq_spatial'):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, ck = load_model(checkpoint, device)
    values = validate_panel(values, panel, ck['vocab'])
    if len(panel) < 3:
        raise ValueError('Fine-tuning requires at least three panel genes')
    splits = check_splits(make_splits(len(values)) if splits is None else splits, len(values))
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    np.savez(dest / 'splits.npz', **splits)
    tensor = torch.tensor(values, device=device)
    lookup = {gene: i for i, gene in enumerate(ck['vocab'])}
    ids = torch.tensor([lookup[g] + 1 for g in panel], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    history, best, best_state = [], float('inf'), None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        metrics = {}
        for scope in ('train', 'val'):
            training = scope == 'train'
            model.train(training)
            rng = np.random.default_rng(seed + epoch if training else seed + 10000)
            rows = splits[scope]
            if training:
                rows = rng.permutation(rows)
            total, batches = 0., 0
            with torch.set_grad_enabled(training):
                for start in range(0, len(rows), 64):
                    batch_rows = rows[start:start + 64]
                    target = tensor[torch.as_tensor(batch_rows, device=device)]
                    mask_array = np.zeros((len(batch_rows), len(panel)), dtype='float32')
                    for i in range(len(batch_rows)):
                        ratio = rng.choice([.25, .5, .75]) if training else .5
                        n_mask = min(len(panel) - 1, max(2, round(len(panel) * ratio)))
                        mask_array[i, rng.choice(len(panel), n_mask, replace=False)] = 1
                    mask = torch.tensor(mask_array, device=device)
                    inputs = target.masked_fill(mask.bool(), 0)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    pred = forward_panel(model, ck, inputs, ids, ids - 1, modality)
                    loss = compute_panel_loss(pred, target, mask, loss_type='huber_corr',
                        huber_delta=1., corr_lambda=.05, corr_min_genes=2)['loss']
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite {scope} loss in epoch {epoch}')
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                    total += float(loss.detach())
                    batches += 1
            metrics[scope] = total / batches
        history.append(dict(epoch=epoch, **metrics, seconds=time.monotonic() - started))
        print(json.dumps(history[-1]), flush=True)
        if metrics['val'] < best:
            best = metrics['val']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        (dest / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if best_state is None:
        raise RuntimeError('No validation checkpoint was produced')
    model.load_state_dict(best_state, strict=True)
    ck = copy.copy(ck)
    ck['model_state_dict'] = best_state
    ck['finetuning'] = dict(panel_genes=list(panel), seed=seed, epochs=epochs,
        batch_size=64, learning_rate=1e-5, weight_decay=1e-5, grad_clip=1.,
        mask_ratios=[.25, .5, .75], validation_mask_ratio=.5, modality=modality,
        best_epoch=min(history, key=lambda x: x['val'])['epoch'],
        splits={k: v.tolist() for k, v in splits.items()})
    torch.save(ck, dest / 'best.pt')
    return model, ck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/impleo.pt'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--splits', type=Path, help='NPZ with train/val/test row vectors, or equivalent JSON')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modality', default='seq_spatial')
    args = parser.parse_args()
    splits = None
    if args.splits:
        if args.splits.suffix == '.npz':
            with np.load(args.splits, allow_pickle=False) as archive:
                splits = {k: archive[k] for k in archive.files}
        else:
            splits = json.loads(args.splits.read_text(encoding='utf-8'))
    finetune(args.checkpoint, np.load(args.expression, allow_pickle=False),
        read_names(args.panel, 'panel_genes'), args.output_dir, splits,
        args.seed, args.epochs, args.device, args.modality)


if __name__ == '__main__':
    main()
