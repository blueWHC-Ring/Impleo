#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz


GROUP_COL_CANDIDATES = [
    "donor_id",
    "patient_id",
    "animal_id",
    "section_id",
    "sample_id",
    "library_id",
    "sample",
    "donor",
    "patient",
    "animal",
    "batch",
    "orig.ident",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--ortholog", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)

    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--split-mode", choices=["within_dataset", "dataset_holdout"], default="dataset_holdout")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-cells-per-gene", type=int, default=20)
    p.add_argument("--hvg-topk", type=int, default=6000)
    p.add_argument("--max-cells-for-hvg", type=int, default=30000)

    p.add_argument("--counts-layer", type=str, default="counts")
    p.add_argument("--min-test-cells", type=int, default=1)

    p.add_argument("--max-estimated-ram-gb", type=float, default=64.0)
    p.add_argument("--sample-cells-for-memory-estimate", type=int, default=2000)

    # Configurable cell-level QC thresholds.
    p.add_argument("--min-counts-per-cell", type=float, default=0.0)
    p.add_argument("--min-genes-per-cell", type=int, default=0)
    p.add_argument("--max-mito-pct", type=float, default=100.0)

    return p.parse_args()


def safe_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def ensure_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return csr_matrix(X)


def normalize_log1p_csr(X: csr_matrix, target_sum: float = 1e4) -> csr_matrix:
    X = ensure_csr(X).astype(np.float32, copy=True)
    if X.nnz > 0:
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        X.data[X.data < 0] = 0.0
        X.eliminate_zeros()

    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib <= 0] = 1.0
    scale = target_sum / lib
    X = sp.diags(scale.astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def load_ortholog_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"shared_gene", "human_symbol", "mouse_symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ortholog TSV 缺列: {sorted(missing)}")

    df = df.copy()
    for col in ["shared_gene", "human_symbol", "mouse_symbol"]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
        df = df[df[col].str.lower() != "nan"]

    if "ortholog_type" in df.columns:
        keep = df["ortholog_type"].astype(str).str.lower().str.contains("one")
        if keep.any():
            df = df[keep].copy()

    df = df.drop_duplicates(subset=["shared_gene", "human_symbol", "mouse_symbol"]).reset_index(drop=True)
    return df


def choose_gene_source_names_from_var(var: pd.DataFrame, var_names: pd.Index) -> pd.Index:
    preferred = [
        "gene_symbol",
        "gene_name",
        "feature_name",
        "symbol",
        "Gene",
        "genes",
    ]
    fallback = pd.Index(var_names.astype(str))

    for col in preferred:
        if col in var.columns:
            s = var[col].astype(str).fillna("").str.strip()
            valid = (s != "") & (s.str.lower() != "nan")
            if valid.mean() >= 0.7:
                values = np.where(valid.to_numpy(), s.to_numpy(), fallback.to_numpy())
                return pd.Index(values.astype(str))

    return fallback


def choose_gene_source_names(adata_backed) -> pd.Index:
    return choose_gene_source_names_from_var(adata_backed.var.copy(), adata_backed.var_names.astype(str))


def choose_raw_gene_source_names(adata_backed) -> Optional[pd.Index]:
    if getattr(adata_backed, "raw", None) is None:
        return None
    raw = adata_backed.raw
    return choose_gene_source_names_from_var(raw.var.copy(), raw.var_names.astype(str))


def build_species_mapping(ortholog_df: pd.DataFrame, species: str) -> Dict[str, str]:
    if species == "human":
        m = ortholog_df[["human_symbol", "shared_gene"]].drop_duplicates("human_symbol")
        return dict(zip(m["human_symbol"].astype(str), m["shared_gene"].astype(str)))
    if species == "mouse":
        m = ortholog_df[["mouse_symbol", "shared_gene"]].drop_duplicates("mouse_symbol")
        return dict(zip(m["mouse_symbol"].astype(str), m["shared_gene"].astype(str)))
    raise ValueError(f"未知 species: {species}")


def map_genes_to_shared(
    source_gene_names: pd.Index,
    species: str,
    ortholog_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    mapping = build_species_mapping(ortholog_df, species)
    shared = source_gene_names.map(mapping)
    keep = ~shared.isna()
    keep_idx = np.where(keep)[0]
    shared_names = shared[keep].astype(str).to_numpy()
    return keep_idx, shared_names


def build_merge_matrix(shared_names: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    unique_genes = pd.Index(shared_names).unique().to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    col_to_new = np.array([gene_to_idx[g] for g in shared_names], dtype=np.int64)

    S = csr_matrix(
        (
            np.ones(len(shared_names), dtype=np.float32),
            (np.arange(len(shared_names), dtype=np.int64), col_to_new),
        ),
        shape=(len(shared_names), len(unique_genes)),
    )
    return S, unique_genes


def merge_duplicate_columns_csr(X: csr_matrix, shared_names: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
    X = ensure_csr(X)
    S, unique_genes = build_merge_matrix(shared_names)
    X2 = (X @ S).tocsr()
    return X2, unique_genes


def choose_group_column(obs: pd.DataFrame) -> Optional[str]:
    n = len(obs)
    if n == 0:
        return None

    for col in GROUP_COL_CANDIDATES:
        if col not in obs.columns:
            continue
        s = obs[col].astype(str).fillna("NA").str.strip()
        nunique = s.nunique(dropna=False)
        if nunique <= 1:
            continue
        if nunique >= int(0.95 * n):
            continue
        return col
    return None


def make_train_test_split(
    obs: pd.DataFrame,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    n = len(obs)
    rng = np.random.default_rng(seed)
    group_col = choose_group_column(obs)

    if group_col is None:
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_ratio)))
        test_idx = idx[:n_test]
        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        return train_mask, test_mask, "random_cells"

    groups = obs[group_col].astype(str).fillna("NA").str.strip().to_numpy()
    group_sizes = pd.Series(groups).value_counts()
    uniq = group_sizes.index.to_numpy()
    rng.shuffle(uniq)

    target_test = max(1, int(round(n * test_ratio)))
    chosen = []
    acc = 0
    for g in uniq:
        chosen.append(g)
        acc += int(group_sizes[g])
        if acc >= target_test:
            break

    chosen = set(chosen)
    test_mask = np.array([g in chosen for g in groups], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask, group_col


def compute_gene_detected_cells(X: csr_matrix) -> np.ndarray:
    X = ensure_csr(X)
    return np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)


def compute_gene_mean(X: csr_matrix) -> np.ndarray:
    X = ensure_csr(X)
    return np.asarray(X.mean(axis=0)).ravel()


def compute_gene_var(X: csr_matrix) -> np.ndarray:
    X = ensure_csr(X).astype(np.float32)
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = mean_sq - mean ** 2
    var[var < 0] = 0.0
    return var


def compute_hvg_rank_fallback(
    X_counts_csr: csr_matrix,
    gene_names: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    X_log = normalize_log1p_csr(X_counts_csr, target_sum=target_sum)

    means = compute_gene_mean(X_log)
    vars_ = compute_gene_var(X_log)

    means = np.nan_to_num(means, nan=0.0, posinf=0.0, neginf=0.0)
    vars_ = np.nan_to_num(vars_, nan=0.0, posinf=0.0, neginf=0.0)

    disp = vars_ / (means + 1e-8)
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.lexsort((-means, -vars_, -disp))
    rank = pd.Index(gene_names)[order].to_numpy()

    del X_log
    gc.collect()
    return pd.Index(rank).unique().to_numpy()


def compute_hvg_rank_on_sample(
    X_counts_csr: csr_matrix,
    gene_names: np.ndarray,
    n_top: int = 6000,
    max_cells: int = 30000,
    seed: int = 42,
) -> np.ndarray:
    X_counts_csr = ensure_csr(X_counts_csr).astype(np.float32)
    n_cells = X_counts_csr.shape[0]

    if X_counts_csr.nnz > 0:
        X_counts_csr.data = np.nan_to_num(
            X_counts_csr.data, nan=0.0, posinf=0.0, neginf=0.0
        )
        X_counts_csr.data[X_counts_csr.data < 0] = 0.0
        X_counts_csr.eliminate_zeros()

    rng = np.random.default_rng(seed)
    if n_cells > max_cells:
        idx = rng.choice(n_cells, size=max_cells, replace=False)
        X_use = X_counts_csr[idx]
    else:
        X_use = X_counts_csr

    try:
        adata = ad.AnnData(X=X_use.copy())
        adata.var_names = pd.Index(gene_names.astype(str))

        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=min(n_top, adata.n_vars),
            inplace=True,
            subset=False,
        )

        hv = adata.var.copy()
        if "highly_variable_rank" in hv.columns:
            hv_top = hv[hv["highly_variable"]].sort_values("highly_variable_rank").index.to_numpy()
            rest = hv[~hv.index.isin(hv_top)].index.to_numpy()
            rank = np.concatenate([hv_top, rest])
        else:
            rank = hv.index.to_numpy()

        del adata
        gc.collect()
        return pd.Index(rank).unique().to_numpy()

    except Exception as e:
        print(f"  [warn] seurat_v3 failed, fallback to sparse dispersion rank: {type(e).__name__}: {e}")

    return compute_hvg_rank_fallback(
        X_counts_csr=X_use,
        gene_names=gene_names,
        target_sum=1e4,
    )


def sanitize_dataset_key(name: str) -> str:
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def estimate_sparse_bytes_from_csr_shape(n_rows: int, est_nnz: float) -> float:
    est_nnz = max(float(est_nnz), 0.0)
    return est_nnz * 8.0 + (n_rows + 1) * 4.0


def _sample_nonzero_values(X, max_values: int = 50000) -> np.ndarray:
    if sp.issparse(X):
        data = np.asarray(X.data, dtype=np.float64)
        if data.size == 0:
            return np.zeros(0, dtype=np.float64)
        if data.size > max_values:
            rng = np.random.default_rng(0)
            idx = rng.choice(data.size, size=max_values, replace=False)
            data = data[idx]
        return data

    arr = np.asarray(X, dtype=np.float64).ravel()
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if arr.size > max_values:
        rng = np.random.default_rng(0)
        idx = rng.choice(arr.size, size=max_values, replace=False)
        arr = arr[idx]
    return arr


def _fraction_integer_like(values: np.ndarray, tol: float = 1e-6) -> float:
    if values.size == 0:
        return 1.0
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals - np.round(vals)) <= tol))


def audit_matrix(X) -> Dict[str, float | bool]:
    if sp.issparse(X):
        if X.nnz > 0:
            min_value = float(np.min(X.data))
            max_value = float(np.max(X.data))
        else:
            min_value = 0.0
            max_value = 0.0
    else:
        arr = np.asarray(X)
        if arr.size > 0:
            min_value = float(np.nanmin(arr))
            max_value = float(np.nanmax(arr))
        else:
            min_value = 0.0
            max_value = 0.0

    sample = _sample_nonzero_values(X)
    frac_integer_like = _fraction_integer_like(sample)

    return {
        "min_value": float(min_value),
        "max_value": float(max_value),
        "frac_integer_like": float(frac_integer_like),
        "has_negative": bool(min_value < 0),
    }


def summarize_name_collisions(names: pd.Index) -> Dict[str, int]:
    s = pd.Index(names.astype(str))
    upper = s.str.upper()
    return {
        "var_name_duplicates": int(s.duplicated(keep=False).sum()),
        "var_name_upper_duplicates": int(upper.duplicated(keep=False).sum()),
    }


@dataclass
class MatrixSourceChoice:
    name: str
    kind: str
    keep_cols: np.ndarray
    shared_names: np.ndarray
    audit: Dict[str, float | bool]
    layer_name: Optional[str] = None


def _get_candidate_layer_names(backed, preferred_counts_layer: str) -> List[str]:
    names: List[str] = []
    for x in [preferred_counts_layer, "counts", "count", "raw_counts", "umi_counts"]:
        if x and x not in names:
            names.append(x)
    return [x for x in names if x in backed.layers]


def _sample_rows(n_obs: int, sample_cells: int, seed: int) -> np.ndarray:
    if n_obs <= 0:
        return np.zeros(0, dtype=np.int64)
    n_take = min(int(sample_cells), int(n_obs))
    rng = np.random.default_rng(seed)
    if n_take == n_obs:
        return np.arange(n_obs, dtype=np.int64)
    return np.sort(rng.choice(n_obs, size=n_take, replace=False))


def choose_matrix_source(
    backed,
    species: str,
    ortholog_df: pd.DataFrame,
    counts_layer: str,
    sample_cells: int,
    seed: int,
) -> Tuple[MatrixSourceChoice, List[Dict[str, object]], Dict[str, int]]:
    x_gene_names = choose_gene_source_names(backed)
    x_keep_cols, x_shared_names = map_genes_to_shared(x_gene_names, species, ortholog_df)
    x_collision = summarize_name_collisions(x_gene_names)

    raw_gene_names = choose_raw_gene_source_names(backed)
    raw_keep_cols = np.zeros(0, dtype=np.int64)
    raw_shared_names = np.zeros(0, dtype=object)
    raw_collision = {"var_name_duplicates": 0, "var_name_upper_duplicates": 0}
    if raw_gene_names is not None:
        raw_keep_cols, raw_shared_names = map_genes_to_shared(raw_gene_names, species, ortholog_df)
        raw_collision = summarize_name_collisions(raw_gene_names)

    row_idx = _sample_rows(backed.n_obs, sample_cells=sample_cells, seed=seed)

    candidates: List[Tuple[Tuple[float, float, float], MatrixSourceChoice]] = []
    audits: List[Dict[str, object]] = []

    for layer_name in _get_candidate_layer_names(backed, counts_layer):
        if len(x_keep_cols) == 0:
            continue
        sub = backed[row_idx, x_keep_cols].to_memory()
        X = _ensure_loaded_matrix_matches_source(sub.layers[layer_name], source=MatrixSourceChoice(name=f"layer:{layer_name}", kind="layer", keep_cols=x_keep_cols, shared_names=x_shared_names, layer_name=layer_name, audit={}), context="choose_matrix_source(layer)")
        audit = audit_matrix(X)
        choice = MatrixSourceChoice(
            name=f"layer:{layer_name}",
            kind="layer",
            keep_cols=x_keep_cols,
            shared_names=x_shared_names,
            layer_name=layer_name,
            audit=audit,
        )
        audits.append({
            "name": choice.name,
            "kind": choice.kind,
            "n_genes_mapped": int(len(choice.shared_names)),
            **audit,
        })
        score = (3.0, 0.0 if not bool(audit["has_negative"]) else -1000.0, float(audit["frac_integer_like"]))
        candidates.append((score, choice))
        del sub, X
        gc.collect()

    if raw_gene_names is not None and len(raw_keep_cols) > 0:
        X = _ensure_loaded_matrix_matches_source(backed.raw[row_idx, raw_keep_cols].X, source=MatrixSourceChoice(name="raw.X", kind="raw", keep_cols=raw_keep_cols, shared_names=raw_shared_names, layer_name=None, audit={}), context="choose_matrix_source(raw)")
        audit = audit_matrix(X)
        choice = MatrixSourceChoice(
            name="raw.X",
            kind="raw",
            keep_cols=raw_keep_cols,
            shared_names=raw_shared_names,
            layer_name=None,
            audit=audit,
        )
        audits.append({
            "name": choice.name,
            "kind": choice.kind,
            "n_genes_mapped": int(len(choice.shared_names)),
            **audit,
        })
        score = (2.0, 0.0 if not bool(audit["has_negative"]) else -1000.0, float(audit["frac_integer_like"]))
        candidates.append((score, choice))
        del X
        gc.collect()

    if len(x_keep_cols) > 0:
        sub = backed[row_idx, x_keep_cols].to_memory()
        X = _ensure_loaded_matrix_matches_source(sub.X, source=MatrixSourceChoice(name="X", kind="X", keep_cols=x_keep_cols, shared_names=x_shared_names, layer_name=None, audit={}), context="choose_matrix_source(X)")
        audit = audit_matrix(X)
        choice = MatrixSourceChoice(
            name="X",
            kind="X",
            keep_cols=x_keep_cols,
            shared_names=x_shared_names,
            layer_name=None,
            audit=audit,
        )
        audits.append({
            "name": choice.name,
            "kind": choice.kind,
            "n_genes_mapped": int(len(choice.shared_names)),
            **audit,
        })
        score = (1.0, 0.0 if not bool(audit["has_negative"]) else -1000.0, float(audit["frac_integer_like"]))
        candidates.append((score, choice))
        del sub, X
        gc.collect()

    if not candidates:
        raise RuntimeError("no_counts_like_candidate")

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]

    collision_meta = {
        "var_name_duplicates": int(x_collision["var_name_duplicates"]),
        "var_name_upper_duplicates": int(x_collision["var_name_upper_duplicates"]),
        "raw_var_name_duplicates": int(raw_collision["var_name_duplicates"]),
        "raw_var_name_upper_duplicates": int(raw_collision["var_name_upper_duplicates"]),
    }

    return chosen, audits, collision_meta


def _ensure_loaded_matrix_matches_source(
    X,
    source: MatrixSourceChoice,
    context: str = "",
) -> csr_matrix:
    X = ensure_csr(X).astype(np.float32)
    expected = int(len(source.shared_names))
    keep_cols = np.asarray(source.keep_cols, dtype=np.int64)

    if X.shape[1] == expected:
        return X

    # Apply the retained column indices when a backed source returns full columns.
    # The resulting matrix follows the shared_names order.
    if keep_cols.size > 0 and X.shape[1] > int(keep_cols.max()):
        X = X[:, keep_cols].tocsr()

    if X.shape[1] != expected:
        raise ValueError(
            f"{context} loaded matrix column mismatch for source={source.name}: "
            f"got {X.shape[1]} cols, expected {expected} mapped cols, "
            f"len(keep_cols)={len(keep_cols)}"
        )
    return X


def _load_counts_matrix_from_source(backed, row_idx: np.ndarray, source: MatrixSourceChoice) -> csr_matrix:
    if row_idx.size == 0:
        return csr_matrix((0, len(source.keep_cols)), dtype=np.float32)

    if source.kind == "raw":
        X = backed.raw[row_idx, source.keep_cols].X
        return _ensure_loaded_matrix_matches_source(
            X,
            source=source,
            context="_load_counts_matrix_from_source(raw)",
        )

    sub = backed[row_idx, source.keep_cols].to_memory()
    if source.kind == "layer":
        X = sub.layers[source.layer_name]  # type: ignore[index]
    else:
        X = sub.X
    out = _ensure_loaded_matrix_matches_source(
        X,
        source=source,
        context=f"_load_counts_matrix_from_source({source.kind})",
    )
    del sub
    gc.collect()
    return out


def estimate_dataset_ram_gb(
    backed,
    row_idx_groups: List[np.ndarray],
    source: MatrixSourceChoice,
    sample_cells: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)

    total_bytes = 0.0
    for row_idx in row_idx_groups:
        n_rows = len(row_idx)
        if n_rows == 0:
            continue

        n_take = min(sample_cells, n_rows)
        take = np.sort(rng.choice(row_idx, size=n_take, replace=False)) if n_take < n_rows else np.asarray(row_idx)

        X = _load_counts_matrix_from_source(backed, take, source)
        if X.shape[0] == 0:
            est_nnz = 0.0
        else:
            nnz_per_row = float(X.nnz) / float(max(1, X.shape[0]))
            est_nnz = nnz_per_row * float(n_rows)

        total_bytes += estimate_sparse_bytes_from_csr_shape(n_rows, est_nnz)
        del X
        gc.collect()

    safety_multiplier = 4.0
    est_bytes = total_bytes * safety_multiplier
    return est_bytes / (1024 ** 3)


def build_skip_meta(
    dataset_key: str,
    path: Path,
    species: str,
    modality: str,
    reason: str,
    n_obs_total: Optional[int] = None,
    n_vars_total: Optional[int] = None,
    n_genes_mapped_before_filter: Optional[int] = None,
    estimated_ram_gb: Optional[float] = None,
) -> dict:
    meta = {
        "dataset_key": dataset_key,
        "source_path": str(path),
        "species": species,
        "modality": modality,
        "skipped": True,
        "skip_reason": reason,
    }
    if n_obs_total is not None:
        meta["n_obs_total"] = int(n_obs_total)
    if n_vars_total is not None:
        meta["n_vars_total"] = int(n_vars_total)
    if n_genes_mapped_before_filter is not None:
        meta["n_genes_mapped_before_filter"] = int(n_genes_mapped_before_filter)
    if estimated_ram_gb is not None:
        meta["estimated_ram_gb"] = float(estimated_ram_gb)
    return meta


def _compute_cell_qc_mask(
    X_counts: csr_matrix,
    gene_names: np.ndarray,
    min_counts_per_cell: float,
    min_genes_per_cell: int,
    max_mito_pct: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    X_counts = ensure_csr(X_counts).astype(np.float32)

    n_cells = X_counts.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=bool), {
            "n_cells_before_qc": 0,
            "n_cells_after_qc": 0,
            "n_cells_removed_by_qc": 0,
        }

    counts_per_cell = np.asarray(X_counts.sum(axis=1)).ravel()
    genes_per_cell = np.asarray(X_counts.getnnz(axis=1)).ravel()

    keep = np.ones(n_cells, dtype=bool)

    if float(min_counts_per_cell) > 0:
        keep &= counts_per_cell >= float(min_counts_per_cell)

    if int(min_genes_per_cell) > 0:
        keep &= genes_per_cell >= int(min_genes_per_cell)

    mito_mask = np.array([str(g).upper().startswith("MT-") for g in gene_names], dtype=bool)
    mito_present = bool(mito_mask.any())
    mito_removed = 0
    if mito_present and float(max_mito_pct) < 100.0:
        mito_counts = np.asarray(X_counts[:, mito_mask].sum(axis=1)).ravel()
        denom = np.maximum(counts_per_cell, 1.0)
        mito_pct = 100.0 * mito_counts / denom
        before = keep.sum()
        keep &= mito_pct <= float(max_mito_pct)
        mito_removed = int(before - keep.sum())

    qc_meta = {
        "n_cells_before_qc": int(n_cells),
        "n_cells_after_qc": int(keep.sum()),
        "n_cells_removed_by_qc": int(n_cells - keep.sum()),
        "min_counts_per_cell": float(min_counts_per_cell),
        "min_genes_per_cell": int(min_genes_per_cell),
        "max_mito_pct": float(max_mito_pct),
        "mito_genes_present": bool(mito_present),
        "mito_qc_removed_cells": int(mito_removed),
    }
    return keep, qc_meta


def process_one_dataset(
    row: pd.Series,
    ortholog_df: pd.DataFrame,
    outdir: Path,
    target_sum: float,
    test_ratio: float,
    seed: int,
    min_train_cells_per_gene: int,
    hvg_topk: int,
    max_cells_for_hvg: int,
    counts_layer: str,
    min_test_cells: int,
    max_estimated_ram_gb: float,
    sample_cells_for_memory_estimate: int,
    split_mode: str,
    min_counts_per_cell: float,
    min_genes_per_cell: int,
    max_mito_pct: float,
) -> dict:
    path = Path(row["path"])
    species = str(row["species"]).strip().lower()
    modality = str(row.get("modality", "")).strip().lower()
    dataset_key = sanitize_dataset_key(row.get("dataset_key", row.get("name", path.stem)))

    print(f"\n[dataset] {dataset_key}")
    print(f"  path: {path}")
    print(f"  species: {species} | modality: {modality}")

    ds_dir = outdir / "datasets" / dataset_key
    meta_path = ds_dir / "meta.json"
    if meta_path.exists():
        print("  skip: already processed")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ds_dir.mkdir(parents=True, exist_ok=True)
    backed = ad.read_h5ad(path, backed="r")

    try:
        obs = backed.obs.copy()
        n_obs_total = backed.n_obs
        n_vars_total = backed.n_vars

        source_choice, source_audits, collision_meta = choose_matrix_source(
            backed=backed,
            species=species,
            ortholog_df=ortholog_df,
            counts_layer=counts_layer,
            sample_cells=sample_cells_for_memory_estimate,
            seed=seed,
        )

        n_genes_mapped_before_filter = int(len(pd.Index(source_choice.shared_names).unique()))
        if n_genes_mapped_before_filter == 0:
            meta = build_skip_meta(
                dataset_key=dataset_key,
                path=path,
                species=species,
                modality=modality,
                reason="no_genes_mapped_to_shared_gene",
                n_obs_total=n_obs_total,
                n_vars_total=n_vars_total,
                n_genes_mapped_before_filter=0,
            )
            meta.update(collision_meta)
            meta["matrix_source_candidates"] = source_audits
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print("  skip: no genes mapped to shared_gene")
            return meta

        if split_mode == "dataset_holdout":
            full_idx = np.arange(len(obs), dtype=np.int64)
            estimated_ram_gb = estimate_dataset_ram_gb(
                backed=backed,
                row_idx_groups=[full_idx],
                source=source_choice,
                sample_cells=sample_cells_for_memory_estimate,
                seed=seed,
            )
            print(f"  estimated_ram_gb={estimated_ram_gb:.2f}")

            if estimated_ram_gb > max_estimated_ram_gb:
                meta = build_skip_meta(
                    dataset_key=dataset_key,
                    path=path,
                    species=species,
                    modality=modality,
                    reason=f"estimated_ram_gb>{max_estimated_ram_gb}",
                    n_obs_total=n_obs_total,
                    n_vars_total=n_vars_total,
                    n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                    estimated_ram_gb=estimated_ram_gb,
                )
                meta.update(collision_meta)
                meta["split_by"] = "dataset_holdout_full"
                meta["n_train"] = int(len(obs))
                meta["n_test"] = 0
                meta["matrix_source_used"] = source_choice.name
                meta["counts_layer_used"] = source_choice.name
                meta["matrix_source_candidates"] = source_audits
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"  skip: estimated RAM {estimated_ram_gb:.2f} GB > {max_estimated_ram_gb:.2f} GB")
                return meta

            X_full_counts = _load_counts_matrix_from_source(backed, full_idx, source_choice)
            X_full_counts, genes = merge_duplicate_columns_csr(X_full_counts, source_choice.shared_names)

            keep_cells, qc_meta = _compute_cell_qc_mask(
                X_counts=X_full_counts,
                gene_names=genes,
                min_counts_per_cell=min_counts_per_cell,
                min_genes_per_cell=min_genes_per_cell,
                max_mito_pct=max_mito_pct,
            )
            if keep_cells.sum() == 0:
                meta = build_skip_meta(
                    dataset_key=dataset_key,
                    path=path,
                    species=species,
                    modality=modality,
                    reason="all_cells_removed_by_qc",
                    n_obs_total=n_obs_total,
                    n_vars_total=n_vars_total,
                    n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                    estimated_ram_gb=estimated_ram_gb,
                )
                meta.update(collision_meta)
                meta.update(qc_meta)
                meta["split_by"] = "dataset_holdout_full"
                meta["n_train"] = 0
                meta["n_test"] = 0
                meta["matrix_source_used"] = source_choice.name
                meta["counts_layer_used"] = source_choice.name
                meta["matrix_source_candidates"] = source_audits
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print("  skip: all cells removed by QC")
                return meta

            X_full_counts = X_full_counts[keep_cells].tocsr()
            obs = obs.iloc[np.where(keep_cells)[0]].reset_index(drop=True)

            keep_gene = compute_gene_detected_cells(X_full_counts) >= int(min_train_cells_per_gene)
            if keep_gene.sum() == 0:
                meta = build_skip_meta(
                    dataset_key=dataset_key,
                    path=path,
                    species=species,
                    modality=modality,
                    reason="no_genes_left_after_min_train_cells_per_gene_filter",
                    n_obs_total=n_obs_total,
                    n_vars_total=n_vars_total,
                    n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                    estimated_ram_gb=estimated_ram_gb,
                )
                meta.update(collision_meta)
                meta.update(qc_meta)
                meta["split_by"] = "dataset_holdout_full"
                meta["n_train"] = int(len(obs))
                meta["n_test"] = 0
                meta["matrix_source_used"] = source_choice.name
                meta["counts_layer_used"] = source_choice.name
                meta["matrix_source_candidates"] = source_audits
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print("  skip: no genes left after min_train_cells_per_gene filter")
                return meta

            X_full_counts = X_full_counts[:, keep_gene]
            genes = genes[keep_gene]

            hvg_rank = compute_hvg_rank_on_sample(
                X_counts_csr=X_full_counts,
                gene_names=genes,
                n_top=hvg_topk,
                max_cells=max_cells_for_hvg,
                seed=seed,
            )

            X_full_log1p = normalize_log1p_csr(X_full_counts, target_sum=target_sum)

            save_npz(ds_dir / "full_log1p.npz", X_full_log1p)
            pd.Series(genes, name="shared_gene").to_csv(ds_dir / "genes.tsv", sep="\t", index=False, header=False)
            pd.Series(hvg_rank, name="shared_gene").to_csv(ds_dir / "hvg_rank.tsv", sep="\t", index=False, header=False)
            obs.to_csv(ds_dir / "full_obs.csv.gz", index=False)

            meta = {
                "dataset_key": dataset_key,
                "source_path": str(path),
                "species": species,
                "modality": modality,
                "skipped": False,
                "split_by": "dataset_holdout_full",
                "n_obs_total": int(n_obs_total),
                "n_vars_total": int(n_vars_total),
                "n_genes_mapped_before_filter": n_genes_mapped_before_filter,
                "n_genes_after_filter": int(len(genes)),
                "n_train": int(len(obs)),
                "n_test": 0,
                "estimated_ram_gb": float(estimated_ram_gb),
                "counts_layer_used": source_choice.name,
                "matrix_source_used": source_choice.name,
                "matrix_source_candidates": source_audits,
                "target_sum": float(target_sum),
                "min_train_cells_per_gene": int(min_train_cells_per_gene),
                "hvg_topk": int(hvg_topk),
                "max_cells_for_hvg": int(max_cells_for_hvg),
                "has_raw": bool(getattr(backed, "raw", None) is not None),
                "max_meaningful_panel_size": int(max(0, len(genes) - 8)),
                **collision_meta,
                **qc_meta,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"  done: n_full={meta['n_train']} genes={meta['n_genes_after_filter']}")
            return meta

        train_mask, test_mask, split_by = make_train_test_split(obs, test_ratio=test_ratio, seed=seed)

        if train_mask.sum() == 0:
            meta = build_skip_meta(
                dataset_key=dataset_key,
                path=path,
                species=species,
                modality=modality,
                reason="empty_train_split",
                n_obs_total=n_obs_total,
                n_vars_total=n_vars_total,
                n_genes_mapped_before_filter=n_genes_mapped_before_filter,
            )
            meta.update(collision_meta)
            meta["matrix_source_candidates"] = source_audits
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print("  skip: empty train split")
            return meta

        if test_mask.sum() < min_test_cells:
            idx_train = np.where(train_mask)[0]
            n_test = max(min_test_cells, int(round(len(obs) * test_ratio)))
            n_test = min(n_test, max(1, len(obs) - 1))
            rng = np.random.default_rng(seed)
            rng.shuffle(idx_train)
            promote = idx_train[:n_test]
            test_mask = np.zeros(len(obs), dtype=bool)
            test_mask[promote] = True
            train_mask = ~test_mask
            split_by = "random_cells_fallback"

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        estimated_ram_gb = estimate_dataset_ram_gb(
            backed=backed,
            row_idx_groups=[train_idx, test_idx],
            source=source_choice,
            sample_cells=sample_cells_for_memory_estimate,
            seed=seed,
        )
        print(f"  estimated_ram_gb={estimated_ram_gb:.2f}")

        if estimated_ram_gb > max_estimated_ram_gb:
            meta = build_skip_meta(
                dataset_key=dataset_key,
                path=path,
                species=species,
                modality=modality,
                reason=f"estimated_ram_gb>{max_estimated_ram_gb}",
                n_obs_total=n_obs_total,
                n_vars_total=n_vars_total,
                n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                estimated_ram_gb=estimated_ram_gb,
            )
            meta.update(collision_meta)
            meta["split_by"] = split_by
            meta["n_train"] = int(train_mask.sum())
            meta["n_test"] = int(test_mask.sum())
            meta["matrix_source_used"] = source_choice.name
            meta["counts_layer_used"] = source_choice.name
            meta["matrix_source_candidates"] = source_audits
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  skip: estimated RAM {estimated_ram_gb:.2f} GB > {max_estimated_ram_gb:.2f} GB")
            return meta

        X_train_counts = _load_counts_matrix_from_source(backed, train_idx, source_choice)
        X_test_counts = _load_counts_matrix_from_source(backed, test_idx, source_choice)

        X_train_counts, genes = merge_duplicate_columns_csr(X_train_counts, source_choice.shared_names)
        X_test_counts, genes_test = merge_duplicate_columns_csr(X_test_counts, source_choice.shared_names)

        if not np.array_equal(genes, genes_test):
            idx_train_g = pd.Index(genes)
            idx_test_g = pd.Index(genes_test)
            common = idx_train_g.intersection(idx_test_g)
            X_train_counts = X_train_counts[:, idx_train_g.get_indexer(common)]
            X_test_counts = X_test_counts[:, idx_test_g.get_indexer(common)]
            genes = common.to_numpy()

        train_keep_cells, train_qc_meta = _compute_cell_qc_mask(
            X_counts=X_train_counts,
            gene_names=genes,
            min_counts_per_cell=min_counts_per_cell,
            min_genes_per_cell=min_genes_per_cell,
            max_mito_pct=max_mito_pct,
        )
        test_keep_cells, test_qc_meta = _compute_cell_qc_mask(
            X_counts=X_test_counts,
            gene_names=genes,
            min_counts_per_cell=min_counts_per_cell,
            min_genes_per_cell=min_genes_per_cell,
            max_mito_pct=max_mito_pct,
        )

        if train_keep_cells.sum() == 0:
            meta = build_skip_meta(
                dataset_key=dataset_key,
                path=path,
                species=species,
                modality=modality,
                reason="all_train_cells_removed_by_qc",
                n_obs_total=n_obs_total,
                n_vars_total=n_vars_total,
                n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                estimated_ram_gb=estimated_ram_gb,
            )
            meta.update(collision_meta)
            meta["split_by"] = split_by
            meta["n_train"] = 0
            meta["n_test"] = int(test_keep_cells.sum())
            meta["matrix_source_used"] = source_choice.name
            meta["counts_layer_used"] = source_choice.name
            meta["matrix_source_candidates"] = source_audits
            meta["train_qc"] = train_qc_meta
            meta["test_qc"] = test_qc_meta
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print("  skip: all train cells removed by QC")
            return meta

        X_train_counts = X_train_counts[train_keep_cells].tocsr()
        X_test_counts = X_test_counts[test_keep_cells].tocsr()

        train_obs = obs.iloc[train_idx].iloc[np.where(train_keep_cells)[0]].reset_index(drop=True)
        test_obs = obs.iloc[test_idx].iloc[np.where(test_keep_cells)[0]].reset_index(drop=True)

        keep_gene = compute_gene_detected_cells(X_train_counts) >= int(min_train_cells_per_gene)
        if keep_gene.sum() == 0:
            meta = build_skip_meta(
                dataset_key=dataset_key,
                path=path,
                species=species,
                modality=modality,
                reason="no_genes_left_after_min_train_cells_per_gene_filter",
                n_obs_total=n_obs_total,
                n_vars_total=n_vars_total,
                n_genes_mapped_before_filter=n_genes_mapped_before_filter,
                estimated_ram_gb=estimated_ram_gb,
            )
            meta.update(collision_meta)
            meta["split_by"] = split_by
            meta["n_train"] = int(train_obs.shape[0])
            meta["n_test"] = int(test_obs.shape[0])
            meta["matrix_source_used"] = source_choice.name
            meta["counts_layer_used"] = source_choice.name
            meta["matrix_source_candidates"] = source_audits
            meta["train_qc"] = train_qc_meta
            meta["test_qc"] = test_qc_meta
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print("  skip: no genes left after min_train_cells_per_gene filter")
            return meta

        X_train_counts = X_train_counts[:, keep_gene]
        X_test_counts = X_test_counts[:, keep_gene]
        genes = genes[keep_gene]

        hvg_rank = compute_hvg_rank_on_sample(
            X_counts_csr=X_train_counts,
            gene_names=genes,
            n_top=hvg_topk,
            max_cells=max_cells_for_hvg,
            seed=seed,
        )

        X_train_log1p = normalize_log1p_csr(X_train_counts, target_sum=target_sum)
        X_test_log1p = normalize_log1p_csr(X_test_counts, target_sum=target_sum)

        save_npz(ds_dir / "train_log1p.npz", X_train_log1p)
        save_npz(ds_dir / "test_log1p.npz", X_test_log1p)

        pd.Series(genes, name="shared_gene").to_csv(ds_dir / "genes.tsv", sep="\t", index=False, header=False)
        pd.Series(hvg_rank, name="shared_gene").to_csv(ds_dir / "hvg_rank.tsv", sep="\t", index=False, header=False)

        train_obs.to_csv(ds_dir / "train_obs.csv.gz", index=False)
        test_obs.to_csv(ds_dir / "test_obs.csv.gz", index=False)

        meta = {
            "dataset_key": dataset_key,
            "source_path": str(path),
            "species": species,
            "modality": modality,
            "skipped": False,
            "split_by": split_by,
            "n_obs_total": int(n_obs_total),
            "n_vars_total": int(n_vars_total),
            "n_genes_mapped_before_filter": int(n_genes_mapped_before_filter),
            "n_genes_after_filter": int(len(genes)),
            "n_train": int(train_obs.shape[0]),
            "n_test": int(test_obs.shape[0]),
            "estimated_ram_gb": float(estimated_ram_gb),
            "counts_layer_used": source_choice.name,
            "matrix_source_used": source_choice.name,
            "matrix_source_candidates": source_audits,
            "target_sum": float(target_sum),
            "min_train_cells_per_gene": int(min_train_cells_per_gene),
            "hvg_topk": int(hvg_topk),
            "max_cells_for_hvg": int(max_cells_for_hvg),
            "has_raw": bool(getattr(backed, "raw", None) is not None),
            "max_meaningful_panel_size": int(max(0, len(genes) - 8)),
            **collision_meta,
            "train_qc": train_qc_meta,
            "test_qc": test_qc_meta,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(
            f"  done: n_train={meta['n_train']} n_test={meta['n_test']} "
            f"genes={meta['n_genes_after_filter']}"
        )
        return meta

    finally:
        if hasattr(backed, "file") and backed.file is not None:
            backed.file.close()
        gc.collect()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    ortholog_df = load_ortholog_table(args.ortholog)

    required_manifest_cols = {"path", "species"}
    missing = required_manifest_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest 缺列: {sorted(missing)}")

    if "use_for_training" in manifest.columns:
        manifest = manifest[manifest["use_for_training"].map(safe_bool)].copy()

    if "modality" in manifest.columns:
        manifest = manifest[manifest["modality"].isin(["rna_seq", "seq_spatial"])].copy()

    if "format" in manifest.columns:
        manifest = manifest[manifest["format"].fillna("h5ad") == "h5ad"].copy()

    manifest = manifest.drop_duplicates(subset=["path"]).reset_index(drop=True)
    if len(manifest) == 0:
        raise RuntimeError("manifest 过滤后没有可处理的数据集")

    print(f"[info] datasets to process: {len(manifest)}")

    dataset_metas: List[dict] = []
    genes_present = set()

    ortholog_order = pd.Index(ortholog_df["shared_gene"].astype(str)).drop_duplicates().tolist()

    for _, row in manifest.iterrows():
        meta = process_one_dataset(
            row=row,
            ortholog_df=ortholog_df,
            outdir=outdir,
            target_sum=args.target_sum,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_train_cells_per_gene=args.min_train_cells_per_gene,
            hvg_topk=args.hvg_topk,
            max_cells_for_hvg=args.max_cells_for_hvg,
            counts_layer=args.counts_layer,
            min_test_cells=args.min_test_cells,
            max_estimated_ram_gb=args.max_estimated_ram_gb,
            sample_cells_for_memory_estimate=args.sample_cells_for_memory_estimate,
            split_mode=args.split_mode,
            min_counts_per_cell=args.min_counts_per_cell,
            min_genes_per_cell=args.min_genes_per_cell,
            max_mito_pct=args.max_mito_pct,
        )
        dataset_metas.append(meta)

        if not meta.get("skipped", False):
            genes_path = outdir / "datasets" / meta["dataset_key"] / "genes.tsv"
            if genes_path.exists():
                ds_genes = pd.read_csv(genes_path, sep="\t", header=None)[0].astype(str).tolist()
                genes_present.update(ds_genes)

        gc.collect()

    global_genes = [g for g in ortholog_order if g in genes_present]

    vocab_dir = outdir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(global_genes, name="shared_gene").to_csv(
        vocab_dir / "global_genes.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    with open(outdir / "gene_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"genes": global_genes}, f, ensure_ascii=False, indent=2)

    registry = pd.DataFrame(dataset_metas)
    registry.to_csv(outdir / "dataset_registry.csv", index=False)

    n_skipped = int(sum(bool(m.get("skipped", False)) for m in dataset_metas))
    summary = {
        "n_datasets": int(len(dataset_metas)),
        "n_skipped": n_skipped,
        "n_processed": int(len(dataset_metas) - n_skipped),
        "n_global_genes": int(len(global_genes)),
        "target_sum": float(args.target_sum),
        "split_mode": str(args.split_mode),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
        "min_train_cells_per_gene": int(args.min_train_cells_per_gene),
        "hvg_topk": int(args.hvg_topk),
        "max_cells_for_hvg": int(args.max_cells_for_hvg),
        "counts_layer": args.counts_layer,
        "max_estimated_ram_gb": float(args.max_estimated_ram_gb),
        "sample_cells_for_memory_estimate": int(args.sample_cells_for_memory_estimate),
        "min_counts_per_cell": float(args.min_counts_per_cell),
        "min_genes_per_cell": int(args.min_genes_per_cell),
        "max_mito_pct": float(args.max_mito_pct),
    }
    with open(outdir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
