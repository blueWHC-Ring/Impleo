"""Load an Impleo checkpoint and predict expression from a named gene panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import PanelCompletionModel


def read_names(path, key=None):
    path = Path(path)
    if path.suffix == '.json':
        value = json.loads(path.read_text(encoding='utf-8'))
        value = value[key] if isinstance(value, dict) else value
    else:
        value = path.read_text(encoding='utf-8').splitlines()
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise ValueError('Expected a nonempty gene-name list: ' + str(path))
    if len(value) != len(set(value)):
        raise ValueError('Repeated gene names: ' + str(path))
    return value


def load_model(checkpoint, device='cuda'):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    if cfg.get('use_dataset_emb', False):
        raise ValueError('Panel prediction requires a checkpoint without dataset embeddings')
    if len(ck['vocab']) != len(set(ck['vocab'])):
        raise ValueError('Repeated checkpoint vocabulary entries')
    keys = ['d_model', 'nhead', 'num_layers', 'dropout', 'use_dataset_emb',
            'use_modality_emb', 'decoder_type', 'decoder_query_chunk_size',
            'decoder_use_concat_query']
    model = PanelCompletionModel(
        num_input_genes=len(ck['vocab']), num_output_genes=len(ck['vocab']),
        num_datasets=max(1, len(ck['dataset2id'])),
        num_modalities=max(1, len(ck['modality2id'])),
        **{key: cfg[key] for key in keys})
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.to(device), ck


def validate_panel(values, panel, vocab):
    values = np.asarray(values, dtype='float32')
    if values.ndim != 2 or not len(values) or values.shape[1] != len(panel):
        raise ValueError('Expression must have shape cells x panel genes')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expression must be finite and nonnegative')
    if not panel or len(panel) != len(set(panel)) or set(panel) - set(vocab):
        raise ValueError('Panel must contain unique checkpoint gene names')
    return values


def forward_panel(model, ck, values, ids, targets, modality='seq_spatial'):
    return model(
        panel_gene_ids=ids[None].expand(len(values), -1), panel_expr=values,
        panel_mask=torch.ones_like(values, dtype=torch.long),
        modality_ids=torch.full((len(values),), ck['modality2id'][modality],
                                device=values.device, dtype=torch.long),
        target_gene_ids=targets)


@torch.inference_mode()
def predict(model, ck, values, panel, targets, batch_size=64, modality='seq_spatial'):
    values = validate_panel(values, panel, ck['vocab'])
    if not targets or len(targets) != len(set(targets)) or set(targets) - set(ck['vocab']):
        raise ValueError('Targets must contain unique checkpoint gene names')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = next(model.parameters()).device
    lookup = {gene: i for i, gene in enumerate(ck['vocab'])}
    ids = torch.tensor([lookup[g] + 1 for g in panel], device=device)
    queries = torch.tensor([lookup[g] for g in targets], device=device)
    model.eval()
    result = np.empty((len(values), len(targets)), dtype='float32')
    for start in range(0, len(values), batch_size):
        x = torch.tensor(values[start:start + batch_size], device=device)
        result[start:start + len(x)] = forward_panel(model, ck, x, ids, queries, modality).float().cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite predictions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expression', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/impleo.pt'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modality', default='seq_spatial')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.out.suffix != '.npy' or args.out.exists():
        raise ValueError('Choose a new .npy output file')
    model, ck = load_model(args.checkpoint, args.device)
    result = predict(model, ck, np.load(args.expression, allow_pickle=False),
                     read_names(args.panel, 'panel_genes'), read_names(args.targets, 'hidden_genes'),
                     args.batch_size, args.modality)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, result)
    print(f'Saved {result.shape} predictions to {args.out}')


if __name__ == '__main__':
    main()
