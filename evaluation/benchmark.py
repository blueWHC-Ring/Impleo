"""Evaluate Impleo and aligned external predictions on identical cells and genes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from impleo.inference import load_model, predict, read_names
from .metrics import compare


def input_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='float32').tobytes()).hexdigest()


def save_prediction(path, prediction, genes, cell_ids, panel, values):
    np.savez_compressed(path, prediction=prediction, genes=np.asarray(genes, dtype=str),
        cell_ids=np.asarray(cell_ids, dtype=str), panel_genes=np.asarray(panel, dtype=str),
        input_sha256=np.asarray(input_digest(values)))


def load_external(path, genes, cell_ids, panel, values):
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in [('genes', genes), ('cell_ids', cell_ids), ('panel_genes', panel)]:
            if not np.array_equal(archive[key].astype(str), np.asarray(expected, dtype=str)):
                raise ValueError(f'{path}: {key} differs from the benchmark order')
        if str(archive['input_sha256'].item()) != input_digest(values):
            raise ValueError(f'{path}: input expression checksum differs')
        result = archive['prediction']
        if result.shape != (len(cell_ids), len(genes)):
            raise ValueError(f'{path}: prediction dimensions differ')
        if np.isinf(result).any():
            raise ValueError(f'{path}: infinite predictions')
        return result


def run(checkpoint, values, panel, targets, truth, cell_ids, output_dir,
        finetuned_checkpoint=None, external=None, device='cuda', modality='seq_spatial'):
    values, truth = np.asarray(values, dtype='float32'), np.asarray(truth, dtype='float32')
    cell_ids = np.asarray(cell_ids, dtype=str)
    if cell_ids.ndim != 1 or len(cell_ids) != len(values) or len(set(cell_ids)) != len(cell_ids):
        raise ValueError('Provide one unique cell identifier per expression row')
    if set(panel) & set(targets):
        raise ValueError('Input panel and predicted benchmark genes must be disjoint')
    if truth.shape != (len(values), len(targets)) or not np.isfinite(truth).all():
        raise ValueError('Truth must be a finite cells x target genes matrix')
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    predictions = {}
    base_vocab = None
    for name, path in [('Impleo zero-shot', checkpoint), ('Impleo fine-tuned', finetuned_checkpoint)]:
        if path is None:
            continue
        model, ck = load_model(path, device)
        if base_vocab is None:
            base_vocab, base_modalities = ck['vocab'], ck['modality2id']
        elif ck['vocab'] != base_vocab or ck['modality2id'] != base_modalities:
            raise ValueError('Base and fine-tuned checkpoint vocabularies/mappings differ')
        predictions[name] = predict(model, ck, values, panel, targets, modality=modality)
        save_prediction(dest / (name.replace(' ', '_') + '.npz'), predictions[name], targets, cell_ids, panel, values)
        del model, ck
    for name, path in (external or {}).items():
        if name in predictions:
            raise ValueError('Repeated method name: ' + name)
        predictions[name] = load_external(path, targets, cell_ids, panel, values)
    scores, summary = compare(predictions, truth, targets)
    scores.to_csv(dest / 'gene_scores.csv')
    summary.to_csv(dest / 'summary.csv')
    (dest / 'condition.json').write_text(json.dumps(dict(cells=len(values), panel_genes=panel,
        target_genes=targets, common_genes=len(scores), input_sha256=input_digest(values),
        methods=list(predictions), modality=modality), indent=2), encoding='utf-8')
    print(summary.to_string())
    return scores, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['expression', 'panel', 'targets', 'truth', 'cell-ids', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/impleo.pt'))
    parser.add_argument('--finetuned-checkpoint', type=Path)
    parser.add_argument('--external', action='append', default=[], metavar='METHOD=FILE.npz')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modality', default='seq_spatial')
    args = parser.parse_args()
    external = {}
    for item in args.external:
        name, path = item.split('=', 1)
        if not name or name in external:
            raise ValueError('Use unique external method names')
        external[name] = Path(path)
    run(args.checkpoint, np.load(args.expression, allow_pickle=False),
        read_names(args.panel, 'panel_genes'), read_names(args.targets, 'hidden_genes'),
        np.load(args.truth, allow_pickle=False), np.load(args.cell_ids, allow_pickle=False),
        args.output_dir, args.finetuned_checkpoint, external, args.device, args.modality)


if __name__ == '__main__':
    main()
