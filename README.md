# Impleo

Impleo predicts gene expression from measured spatial transcriptomics panels using a Transformer encoder and target-gene cross-attention decoder.

## Repository

```text
impleo/         Model, preprocessing, data loading, loss and prediction
training/       Pretraining and one panel-masking fine-tuning workflow
evaluation/     Gene Pearson benchmark and comparison plots
examples/       Three executable Jupyter notebooks
data/           Downloader for example and pretraining data
checkpoints/    Pretrained Impleo checkpoint
```

## Installation

Python 3.10 or later is required. Install a PyTorch build appropriate for your CPU or CUDA environment, then install Impleo from the repository root:

```bash
git clone https://github.com/blueWHC-Ring/Impleo.git
cd Impleo
python -m pip install -e .
```

The notebooks select CUDA when available and otherwise use CPU. Training on the complete pretraining collection requires a CUDA GPU. Plotting uses Arial when installed.

## Examples

```bash
python -m data.download --packs example
jupyter lab examples
```

Run these notebooks in order:

1. [Prediction](examples/01_prediction.ipynb): predict unmeasured genes in the STARmap example.
2. [Fine-tuning](examples/02_finetuning.ipynb): adapt Impleo using the measured panel.
3. [Benchmark](examples/03_benchmark.ipynb): compare predictions and draw bar, box/scatter and method-pair diagonal plots.

The example contains all 1,549 cells, a 128-gene panel (version 0), and normalized target expression. Data downloads are versioned and checked with SHA-256. Additional data and source citations are available in [Impleo-data](https://github.com/blueWHC-Ring/Impleo-data).

## Prediction

Inputs are a cells-by-panel NumPy matrix and ordered gene-name lists. Use checkpoint-compatible normalized expression: mapped assay counts are normalized to a total of 10,000 per cell and transformed with `log1p` before selecting the input panel. The prediction command uses the supplied values unchanged.

```bash
python -m impleo.inference \
  --expression data/assets/starmap_p128_v0/input_panel.npy \
  --panel data/assets/starmap_p128_v0/protocol.json \
  --targets data/assets/starmap_p128_v0/hidden_genes.json \
  --checkpoint checkpoints/impleo.pt \
  --out results/prediction.npy --device cuda
```

Gene files may contain one name per line or a JSON list. The example protocol JSON contains `panel_genes`. The checkpoint contains model settings, the ordered gene vocabulary, and modality mappings; loading requires no original server paths.

## Pretraining

Download the processed training collection (approximately 8.2 GB):

```bash
python -m data.download --packs training
python -m training.pretrain \
  --processed-dir data/assets/pretraining \
  --output-dir results/pretraining --device cuda
```

Default settings match the supplied checkpoint configuration: 6 epochs, batch size 64, learning rate `1e-4`, panels 98/128/256/512, 5 panel versions, 256-dimensional embeddings and 4 encoder layers. The dataset membership is read from the bundled checkpoint. For another collection, supply `--dataset-split-file` with `train` and `val` dataset-key lists, or use `--train-datasets` and `--val-datasets`. Validation loss selects `best.pt`. Legacy `test` split keys and `--test-datasets` are accepted for compatibility with existing assets. Use `--help` for configuration options.

To preprocess your own H5AD files:

```bash
python -m impleo.preprocessing \
  --manifest source_manifest.csv --ortholog orthologs.tsv \
  --outdir data/processed --split-mode dataset_holdout
```

The manifest uses `dataset_key`, `path`, `species`, `modality`, `format`, and `use_for_training` columns. Modalities are `rna_seq` or `seq_spatial`. The ortholog TSV contains `shared_gene`, `human_symbol`, and `mouse_symbol`. Preprocessing writes the sparse expression matrices, gene lists, vocabulary and registry consumed by training. Data paths in a custom manifest must point to locally available source files.

## Fine-tuning

```bash
python -m training.finetune \
  --expression data/assets/starmap_p128_v0/input_panel.npy \
  --panel data/assets/starmap_p128_v0/protocol.json \
  --checkpoint checkpoints/impleo.pt \
  --output-dir results/finetuning --device cuda
```

This workflow updates all parameters for 5 epochs with AdamW (`lr=1e-5`, weight decay `1e-5`, batch size 64). Training masks 25%, 50%, or 75% of each cell's panel expression; validation masks 50%. Masked values are set to zero while retaining gene tokens. The loss combines Huber reconstruction with a Pearson term (`0.05`); the minimum validation loss selects `best.pt`.

The default cell split is 70/10/20 (seed 20261024). Supply `--splits` to use predefined `train`/`val`/`test` row-index vectors in NPZ or JSON format. Outputs include the selected checkpoint, exact split indices and training history.

## Evaluation and baselines

The benchmark computes **gene-wise Pearson correlation across cells**, uses all supplied rows, and compares methods on the common finite target genes. The example evaluates all cells; row subsets can be supplied explicitly. Error bars show the standard deviation across matched genes. Plot annotations use paired gene-level Wilcoxon tests with Holm correction.

The comparison baselines are [SpaGE](https://github.com/tabdelaal/SpaGE), [Tangram](https://github.com/broadinstitute/Tangram), and [scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial). Install and run them separately. SpaGE and Tangram use species- and tissue-matched single-cell references. Give every method the same panel values, panel genes, cell order and target genes.

To import a baseline prediction, create an NPZ with the shared input identifiers using this helper:

```python
from evaluation.benchmark import save_prediction

save_prediction("results/spage.npz", prediction, targets, cell_ids, panel, values)
```

The file stores `prediction`, `genes`, `cell_ids`, `panel_genes` and the input matrix checksum. `prediction` must have shape cells × target genes. The benchmark checks those identifiers and the checksum before combining results. Use `NaN` columns for genes outside a baseline's vocabulary; common finite genes are used for all methods.

Notebook 03 accepts these files through its `external` dictionary. The command-line equivalent is `python -m evaluation.benchmark --help`. Plots can also be generated from a saved score table:

```bash
python -m evaluation.plotting \
  --scores results/benchmark/gene_scores.csv \
  --output-dir results/benchmark/figures
```

Use a new output directory for each training or evaluation run. Generated predictions, tables and figures are stored under `results/` and are not part of the source repository.
