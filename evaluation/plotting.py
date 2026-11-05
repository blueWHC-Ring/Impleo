"""Draw method comparisons from matched gene-wise Pearson scores."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def colors(methods):
    gray = iter(plt.cm.Greys(np.linspace(.45, .75, max(1, len(methods)))))
    return ['#2365A4' if 'fine-tuned' in m else '#80B7D8' if m.startswith('Impleo') else next(gray) for m in methods]


def draw(scores, output_dir):
    if scores.empty or not np.isfinite(scores.to_numpy()).all():
        raise ValueError('Use nonempty common finite gene scores')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 15, 'axes.labelsize': 17,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    methods = list(scores)
    palette = colors(methods)
    means, sd = scores.mean().to_numpy(), scores.std().fillna(0).to_numpy()

    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(dest / f'{name}.{ext}', dpi=250, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.1), 5))
    for i, method in enumerate(methods):
        ax.bar(i, means[i], yerr=sd[i], color=palette[i], width=.65, capsize=4,
               edgecolor='white', linewidth=.7, label=method)
    ax.set_xticks([])
    ax.set_ylabel('Gene Pearson correlation')
    ax.spines[['top', 'right']].set_visible(False)
    if 'Impleo fine-tuned' in methods:
        ours = methods.index('Impleo fine-tuned')
        lo, hi = ax.get_ylim()
        step = max(.05, (hi - lo) * .09)
        level = max(means + sd) + step
        # Paired gene-level Wilcoxon tests with Holm adjustment.
        pairs = [i for i in range(len(methods)) if i != ours]
        raw = []
        for i in pairs:
            diff = scores.iloc[:, ours].to_numpy() - scores.iloc[:, i].to_numpy()
            raw.append(1. if np.all(diff == 0) else wilcoxon(diff).pvalue)
        order = np.argsort(raw)
        adjusted, previous = np.ones(len(raw)), 0.
        for rank, j in enumerate(order):
            previous = max(previous, min(1., raw[j] * (len(raw) - rank)))
            adjusted[j] = previous
        for i, p in zip(pairs, adjusted):
            label = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
            ax.plot([i, i, ours, ours], [level, level + step / 4, level + step / 4, level], color='.25', lw=1)
            ax.text((i + ours) / 2, level + step / 4, label, ha='center', va='bottom', fontsize=14)
            level += step
        ax.set_ylim(lo, level + step / 2)
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.01, .5), borderaxespad=0)
    save(fig, 'barplot')

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.3), 5))
    box = ax.boxplot([scores[m].to_numpy() for m in methods], patch_artist=True,
        showfliers=False, widths=.55, medianprops=dict(color='black', linewidth=1.4))
    rng = np.random.default_rng(42)
    for i, (method, color) in enumerate(zip(methods, palette), 1):
        box['boxes'][i - 1].set_facecolor(color)
        box['boxes'][i - 1].set_alpha(.4)
        ax.scatter(i + rng.uniform(-.18, .18, len(scores)), scores[method],
                   color=color, s=10, alpha=.55, linewidths=0)
    ax.set_xticks(np.arange(1, len(methods) + 1), methods, rotation=25, ha='right')
    ax.set_ylabel('Gene Pearson correlation')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'boxplot_gene_points')

    if len(methods) < 2:
        return
    n = len(methods) - 1
    fig, axes = plt.subplots(n, n, figsize=(4.2 * n, 4.2 * n), squeeze=False)
    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if col > row:
                ax.axis('off')
                continue
            x, y = scores.iloc[:, col], scores.iloc[:, row + 1]
            ax.scatter(x, y, c=palette[row + 1], alpha=.6, s=15, linewidths=0)
            ax.plot([-1, 1], [-1, 1], '--', c='.4', lw=1)
            ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal')
            ax.text(.04, .94, f'{100 * (y > x).mean():.1f}% genes higher',
                    transform=ax.transAxes, va='top', fontsize=17)
            ax.set_xlabel(methods[col], fontsize=16)
            ax.set_ylabel(methods[row + 1], fontsize=16)
            ax.set_xticks([-.5, 0, .5, 1])
            ax.set_yticks([-.5, 0, .5, 1])
    fig.tight_layout()
    save(fig, 'all_method_diagonal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    draw(pd.read_csv(args.scores, index_col='gene'), args.output_dir)


if __name__ == '__main__':
    main()
