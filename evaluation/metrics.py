"""Gene-wise Pearson correlation over aligned cells."""
import numpy as np
import pandas as pd


def gene_pearson(prediction, truth):
    prediction, truth = np.asarray(prediction, dtype='float64'), np.asarray(truth, dtype='float64')
    if truth.ndim != 2 or not len(truth) or prediction.shape != truth.shape:
        raise ValueError('Prediction and truth must have identical cells x genes shape')
    x, y = prediction - prediction.mean(0), truth - truth.mean(0)
    den = np.sqrt(np.sum(x * x, 0) * np.sum(y * y, 0))
    return np.divide(np.sum(x * y, 0), den, out=np.full(truth.shape[1], np.nan), where=den > 1e-12)


def compare(predictions, truth, genes):
    if not predictions or len(genes) != truth.shape[1] or len(genes) != len(set(genes)):
        raise ValueError('Provide methods and unique target gene names matching truth')
    scores = pd.DataFrame({method: gene_pearson(pred, truth) for method, pred in predictions.items()}, index=genes)
    scores.index.name = 'gene'
    common = np.isfinite(scores.to_numpy()).all(1)
    if not common.any():
        raise ValueError('No common finite target genes')
    matched = scores.loc[common].copy()
    summary = matched.agg(['mean', 'std', 'count']).T.rename(columns={'count': 'genes'})
    summary.index.name = 'method'
    return matched, summary
