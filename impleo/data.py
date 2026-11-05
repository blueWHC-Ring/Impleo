from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


PAD_GENE_ID = 0


def load_gene_vocab(processed_dir: str | Path) -> List[str]:
    processed_dir = Path(processed_dir)

    gene_vocab_json = processed_dir / "gene_vocab.json"
    global_genes_tsv = processed_dir / "vocab" / "global_genes.tsv"

    if gene_vocab_json.exists():
        with open(gene_vocab_json, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict) and "genes" in obj:
            genes = [str(x).strip() for x in obj["genes"]]
            genes = [g for g in genes if g and g.lower() != "nan"]
            if genes:
                return genes

        if isinstance(obj, dict):
            try:
                items = sorted(obj.items(), key=lambda kv: int(kv[1]))
                genes = [str(k).strip() for k, _ in items]
                genes = [g for g in genes if g and g.lower() != "nan"]
                if genes:
                    return genes
            except Exception:
                pass

        if isinstance(obj, list):
            genes = [str(x).strip() for x in obj]
            genes = [g for g in genes if g and g.lower() != "nan"]
            if genes:
                return genes

    if global_genes_tsv.exists():
        genes = pd.read_csv(global_genes_tsv, sep="\t", header=None)[0].astype(str).tolist()
        genes = [g.strip() for g in genes if str(g).strip() and str(g).strip().lower() != "nan"]
        if genes:
            return genes

    raise FileNotFoundError(f"Neither usable {gene_vocab_json} nor {global_genes_tsv} exists.")


def _read_gene_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    genes: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        g = str(line).strip()
        if g and g.lower() != "nan":
            genes.append(g)
    return genes


def _load_single_split_matrix(ds_dir: Path, split: str) -> sp.csr_matrix:
    # Prefer the normalized log1p expression matrix.
    preferred = [
        ds_dir / f"{split}_log1p.npz",
        ds_dir / f"{split}_X.npz",   # Alternate matrix filenames
    ]
    for p in preferred:
        if p.exists():
            return sp.load_npz(p).tocsr()
    raise FileNotFoundError(
        f"Missing matrix for split={split} under {ds_dir}. "
        f"Tried: {[str(x) for x in preferred]}"
    )


def _load_split_matrix(ds_dir: Path, split: str) -> sp.csr_matrix:
    if split in {"train", "test"}:
        return _load_single_split_matrix(ds_dir, split)

    if split == "full":
        full_candidates = [
            ds_dir / "full_log1p.npz",
            ds_dir / "full_X.npz",   # Alternate matrix filenames
        ]
        for p in full_candidates:
            if p.exists():
                return sp.load_npz(p).tocsr()

        parts: List[sp.csr_matrix] = []
        for part in ("train", "test"):
            try:
                parts.append(_load_single_split_matrix(ds_dir, part))
            except FileNotFoundError:
                continue

        if not parts:
            raise FileNotFoundError(
                f"Missing matrix for split=full under {ds_dir}. "
                "Expected full matrix or at least one of train/test matrices to exist."
            )
        if len(parts) == 1:
            return parts[0]
        return sp.vstack(parts, format="csr")

    raise ValueError(f"Unknown split: {split}. Expected one of train/test/full")


def _stable_seed(*parts: object) -> int:
    text = "||".join(str(x) for x in parts)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _make_unique_ranked_variants(
    ranked_gene_ids: np.ndarray,
    ranked_local_cols: np.ndarray,
    requested_size: int,
    num_versions: int,
    panel_pool_multiplier: float,
    panel_anchor: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(ranked_gene_ids) == 0:
        return [(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)) for _ in range(max(1, num_versions))]

    actual_k = max(1, min(int(requested_size), len(ranked_gene_ids)))
    candidate_pool = min(
        len(ranked_gene_ids),
        max(actual_k, int(math.ceil(actual_k * float(panel_pool_multiplier)))),
    )
    anchor_k = min(int(panel_anchor), actual_k, candidate_pool)

    fixed_idx = np.arange(anchor_k, dtype=np.int64)
    pool_rest = np.arange(anchor_k, candidate_pool, dtype=np.int64)
    need_random = actual_k - anchor_k

    rng = np.random.default_rng(seed)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    seen: set[Tuple[int, ...]] = set()

    max_attempts = max(20, num_versions * 10)
    attempts = 0

    while len(out) < max(1, int(num_versions)) and attempts < max_attempts:
        attempts += 1
        if need_random <= 0 or len(pool_rest) == 0:
            chosen = fixed_idx.copy()
        else:
            take = min(need_random, len(pool_rest))
            sampled = np.sort(rng.choice(pool_rest, size=take, replace=False))
            chosen = np.sort(np.concatenate([fixed_idx, sampled], axis=0))

        key = tuple(int(x) for x in chosen.tolist())
        if key in seen and len(pool_rest) >= need_random and candidate_pool > actual_k:
            continue

        seen.add(key)
        out.append(
            (
                ranked_gene_ids[chosen].astype(np.int64, copy=True),
                ranked_local_cols[chosen].astype(np.int64, copy=True),
            )
        )

    if not out:
        chosen = np.arange(actual_k, dtype=np.int64)
        out.append(
            (
                ranked_gene_ids[chosen].astype(np.int64, copy=True),
                ranked_local_cols[chosen].astype(np.int64, copy=True),
            )
        )

    while len(out) < max(1, int(num_versions)):
        out.append((out[-1][0].copy(), out[-1][1].copy()))

    return out


class ProcessedDatasetView:
    def __init__(
        self,
        dataset_key: str,
        dataset_id: int,
        species: str,
        modality: str,
        modality_id: int,
        X: sp.csr_matrix,
        local_gene_names: List[str],
        local_to_global: np.ndarray,
        ranked_panel_gene_ids: np.ndarray,
        ranked_panel_local_cols: np.ndarray,
        panel_variants: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    ):
        self.dataset_key = dataset_key
        self.dataset_id = int(dataset_id)
        self.species = species
        self.modality = modality
        self.modality_id = int(modality_id)
        self.X = X.tocsr()
        self.local_gene_names = local_gene_names
        self.local_to_global = np.asarray(local_to_global, dtype=np.int64)
        self.ranked_panel_gene_ids = np.asarray(ranked_panel_gene_ids, dtype=np.int64)
        self.ranked_panel_local_cols = np.asarray(ranked_panel_local_cols, dtype=np.int64)
        self.panel_variants = panel_variants
        self.n = int(self.X.shape[0])


class MultiDatasetPanelCompletion(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        panel_sizes: Sequence[int],
        seed: int = 42,
        fixed_panel_size: Optional[int] = None,
        fixed_panel_version: Optional[int] = None,
        num_panel_versions: int = 5,
        panel_pool_multiplier: float = 4.0,
        panel_anchor: int = 32,
        dataset_keys: Optional[Sequence[str]] = None,
        dataset_id_map: Optional[Dict[str, int]] = None,
        modality_id_map: Optional[Dict[str, int]] = None,
        dataset_id_fallback: int = 0,
        modality_id_fallback: int = 0,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.panel_sizes = [int(x) for x in panel_sizes]
        self.fixed_panel_size = fixed_panel_size
        self.fixed_panel_version = fixed_panel_version
        self.seed = int(seed)
        self.num_panel_versions = max(1, int(num_panel_versions))
        self.panel_pool_multiplier = float(panel_pool_multiplier)
        self.panel_anchor = int(panel_anchor)

        self.vocab = load_gene_vocab(self.processed_dir)
        self.gene2col = {g: i for i, g in enumerate(self.vocab)}
        self.gene2id = {g: i + 1 for i, g in enumerate(self.vocab)}  # 0 reserved for PAD

        registry_path = self.processed_dir / "dataset_registry.csv"
        self.dataset_registry = pd.read_csv(registry_path)

        if "skipped" in self.dataset_registry.columns:
            self.dataset_registry = self.dataset_registry[
                ~self.dataset_registry["skipped"].fillna(False).astype(bool)
            ].copy()

        selected_keys: Optional[set[str]] = None
        if dataset_keys is not None:
            selected_keys = {str(x) for x in dataset_keys}
            self.dataset_registry = self.dataset_registry[
                self.dataset_registry["dataset_key"].astype(str).isin(selected_keys)
            ].copy()

        self.dataset_registry = self.dataset_registry.reset_index(drop=True)

        self.datasets: List[ProcessedDatasetView] = []
        self.rowmap: List[Tuple[int, int]] = []

        observed_dataset_ids: Dict[str, int] = {}
        observed_modality_ids: Dict[str, int] = {}
        loaded_modalities: List[str] = []

        requested_panel_sizes = set(int(x) for x in self.panel_sizes)
        if self.fixed_panel_size is not None:
            requested_panel_sizes.add(int(self.fixed_panel_size))

        for _, row in self.dataset_registry.iterrows():
            dataset_key = str(row["dataset_key"])
            ds_dir = self.processed_dir / "datasets" / dataset_key

            if not ds_dir.exists():
                continue

            try:
                X = _load_split_matrix(ds_dir, self.split)
            except FileNotFoundError:
                continue

            genes_path = ds_dir / "genes.tsv"
            local_gene_names = _read_gene_list(genes_path)
            if not local_gene_names:
                continue

            n_cols = X.shape[1]
            if len(local_gene_names) != n_cols:
                m = min(len(local_gene_names), n_cols)
                local_gene_names = local_gene_names[:m]
                X = X[:, :m].tocsr()

            keep_local = [i for i, g in enumerate(local_gene_names) if g in self.gene2col]
            if not keep_local:
                continue

            if len(keep_local) != X.shape[1]:
                X = X[:, keep_local].tocsr()
                local_gene_names = [local_gene_names[i] for i in keep_local]

            local_to_global = np.array([self.gene2col[g] for g in local_gene_names], dtype=np.int64)

            panel_rank_path = None
            if "panel_rank_path" in row and pd.notna(row["panel_rank_path"]):
                candidate = Path(str(row["panel_rank_path"]))
                if candidate.exists():
                    panel_rank_path = candidate

            if panel_rank_path is None:
                candidate = ds_dir / "hvg_rank.tsv"
                if candidate.exists():
                    panel_rank_path = candidate

            if panel_rank_path is not None:
                ranked_panel_genes = _read_gene_list(panel_rank_path)
            else:
                ranked_panel_genes = list(local_gene_names)

            local_gene_set = set(local_gene_names)
            ranked_panel_genes = [g for g in ranked_panel_genes if g in local_gene_set and g in self.gene2id]

            seen = set()
            ranked_panel_genes = [g for g in ranked_panel_genes if not (g in seen or seen.add(g))]

            if not ranked_panel_genes:
                ranked_panel_genes = [g for g in local_gene_names if g in self.gene2id]

            local_col_map = {g: i for i, g in enumerate(local_gene_names)}
            ranked_panel_local_cols = np.array([local_col_map[g] for g in ranked_panel_genes], dtype=np.int64)
            ranked_panel_gene_ids = np.array([self.gene2id[g] for g in ranked_panel_genes], dtype=np.int64)

            panel_variants: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
            for k in sorted(requested_panel_sizes):
                panel_variants[int(k)] = _make_unique_ranked_variants(
                    ranked_gene_ids=ranked_panel_gene_ids,
                    ranked_local_cols=ranked_panel_local_cols,
                    requested_size=int(k),
                    num_versions=self.num_panel_versions,
                    panel_pool_multiplier=self.panel_pool_multiplier,
                    panel_anchor=self.panel_anchor,
                    seed=_stable_seed(self.seed, dataset_key, int(k)),
                )

            modality = str(row.get("modality", ""))

            if dataset_id_map is None:
                dataset_id = len(observed_dataset_ids)
                observed_dataset_ids[dataset_key] = dataset_id
            else:
                dataset_id = int(dataset_id_map.get(dataset_key, dataset_id_fallback))
                observed_dataset_ids[dataset_key] = dataset_id

            if modality_id_map is None:
                if modality not in observed_modality_ids:
                    observed_modality_ids[modality] = len(observed_modality_ids)
                modality_id = observed_modality_ids[modality]
            else:
                modality_id = int(modality_id_map.get(modality, modality_id_fallback))
                observed_modality_ids[modality] = modality_id

            loaded_modalities.append(modality)

            view = ProcessedDatasetView(
                dataset_key=dataset_key,
                dataset_id=dataset_id,
                species=str(row.get("species", "")),
                modality=modality,
                modality_id=modality_id,
                X=X,
                local_gene_names=local_gene_names,
                local_to_global=local_to_global,
                ranked_panel_gene_ids=ranked_panel_gene_ids,
                ranked_panel_local_cols=ranked_panel_local_cols,
                panel_variants=panel_variants,
            )

            ds_idx = len(self.datasets)
            self.datasets.append(view)
            for i in range(view.n):
                self.rowmap.append((ds_idx, i))

        if not self.datasets:
            key_msg = ""
            if selected_keys is not None:
                key_msg = f" | requested dataset_keys={sorted(selected_keys)}"
            raise RuntimeError(
                f"No usable datasets found in {self.processed_dir} for split={self.split}{key_msg}"
            )

        if dataset_id_map is None:
            self.dataset2id = observed_dataset_ids
        else:
            self.dataset2id = {str(k): int(v) for k, v in dataset_id_map.items()}

        if modality_id_map is None:
            uniq_modalities = sorted(set(loaded_modalities))
            self.modality2id = {m: observed_modality_ids[m] for m in uniq_modalities}
        else:
            self.modality2id = {str(k): int(v) for k, v in modality_id_map.items()}

    def __len__(self) -> int:
        return len(self.rowmap)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        ds_idx, row_idx = self.rowmap[idx]
        view = self.datasets[ds_idx]

        x_local = view.X.getrow(row_idx).toarray().astype(np.float32).ravel()

        return {
            "ds_idx": int(ds_idx),
            "dataset_id": torch.tensor(view.dataset_id, dtype=torch.long),
            "modality_id": torch.tensor(view.modality_id, dtype=torch.long),
            "dataset_name": view.dataset_key,
            "x_local": torch.from_numpy(x_local.astype(np.float32)),
            "local_to_global": torch.from_numpy(view.local_to_global.astype(np.int64)),
        }


class PanelBatchCollator:
    def __init__(
        self,
        dataset: MultiDatasetPanelCompletion,
        batch_panel_sampling: bool = True,
        fixed_panel_size: Optional[int] = None,
        fixed_panel_version: Optional[int] = None,
        seed: int = 42,
        min_masked_genes: int = 8,
    ) -> None:
        self.dataset = dataset
        self.batch_panel_sampling = bool(batch_panel_sampling)
        self.fixed_panel_size = fixed_panel_size
        self.fixed_panel_version = fixed_panel_version
        self.min_masked_genes = max(1, int(min_masked_genes))
        self.rng = np.random.default_rng(int(seed))

    def _eligible_panel_sizes(self, view: ProcessedDatasetView) -> List[int]:
        n_local_genes = int(len(view.local_to_global))
        n_ranked_panel_genes = int(len(view.ranked_panel_gene_ids))
        max_panel_genes = min(n_ranked_panel_genes, max(0, n_local_genes - self.min_masked_genes))

        panel_sizes = self.dataset.panel_sizes
        if not panel_sizes:
            raise ValueError("dataset.panel_sizes is empty")

        return [int(p) for p in panel_sizes if int(p) <= max_panel_genes]

    def _choose_panel_size(self, view: ProcessedDatasetView) -> int:
        fixed = self.fixed_panel_size
        if fixed is None:
            fixed = self.dataset.fixed_panel_size

        eligible = self._eligible_panel_sizes(view)

        if fixed is not None:
            fixed = int(fixed)
            if fixed not in eligible:
                raise ValueError(
                    f"fixed_panel_size={fixed} is not valid for dataset={view.dataset_key}: "
                    f"n_local_genes={len(view.local_to_global)}, "
                    f"n_ranked_panel_genes={len(view.ranked_panel_gene_ids)}, "
                    f"min_masked_genes={self.min_masked_genes}, "
                    f"eligible_panel_sizes={eligible}"
                )
            return fixed

        if not eligible:
            raise ValueError(
                f"No valid panel size for dataset={view.dataset_key}: "
                f"n_local_genes={len(view.local_to_global)}, "
                f"n_ranked_panel_genes={len(view.ranked_panel_gene_ids)}, "
                f"requested_panel_sizes={self.dataset.panel_sizes}, "
                f"min_masked_genes={self.min_masked_genes}"
            )

        return int(self.rng.choice(eligible))

    def _choose_panel_version(self, view: ProcessedDatasetView, panel_size: int) -> int:
        variants = view.panel_variants[int(panel_size)]
        if not variants:
            return 0
        if self.fixed_panel_version is not None:
            return int(self.fixed_panel_version) % len(variants)
        if self.dataset.fixed_panel_version is not None:
            return int(self.dataset.fixed_panel_version) % len(variants)
        if self.batch_panel_sampling:
            return int(self.rng.integers(0, len(variants)))
        return 0

    def __call__(self, batch: List[Dict[str, torch.Tensor | str | int]]) -> Dict[str, torch.Tensor | List[str]]:
        if len(batch) == 0:
            raise ValueError("Empty batch")

        chosen_panel_sizes: List[int] = []

        panel_gene_ids_list: List[torch.Tensor] = []
        panel_expr_list: List[torch.Tensor] = []
        dataset_ids_list: List[torch.Tensor] = []
        modality_ids_list: List[torch.Tensor] = []
        dataset_names: List[str] = []
        target_expr_list: List[torch.Tensor] = []
        target_loss_mask_list: List[torch.Tensor] = []

        chosen_versions: List[int] = []

        for item in batch:
            ds_idx = int(item["ds_idx"])
            view = self.dataset.datasets[ds_idx]

            panel_size = self._choose_panel_size(view)
            chosen_panel_sizes.append(panel_size)

            version_idx = self._choose_panel_version(view, panel_size)
            chosen_versions.append(version_idx)

            panel_gene_ids_np, panel_local_cols_np = view.panel_variants[int(panel_size)][version_idx]
            panel_gene_ids = torch.from_numpy(panel_gene_ids_np.astype(np.int64))
            panel_local_cols = torch.from_numpy(panel_local_cols_np.astype(np.int64))

            x_local = item["x_local"]
            local_to_global = item["local_to_global"]

            target_expr = torch.zeros(len(self.dataset.vocab), dtype=torch.float32)
            target_loss_mask = torch.zeros(len(self.dataset.vocab), dtype=torch.float32)

            target_expr[local_to_global] = x_local
            target_loss_mask[local_to_global] = 1.0

            if len(panel_local_cols) > 0:
                panel_expr = x_local[panel_local_cols].to(torch.float32)
                panel_global_cols = panel_gene_ids.to(torch.long) - 1
                target_loss_mask[panel_global_cols] = 0.0
            else:
                panel_expr = torch.zeros(0, dtype=torch.float32)

            panel_gene_ids_list.append(panel_gene_ids.to(torch.long))
            panel_expr_list.append(panel_expr)
            dataset_ids_list.append(item["dataset_id"])
            modality_ids_list.append(item["modality_id"])
            dataset_names.append(str(item["dataset_name"]))
            target_expr_list.append(target_expr)
            target_loss_mask_list.append(target_loss_mask)

        bsz = len(batch)
        max_len = max((len(x) for x in panel_gene_ids_list), default=0)
        if max_len == 0:
            max_len = 1

        gene_ids = torch.full((bsz, max_len), PAD_GENE_ID, dtype=torch.long)
        expr = torch.zeros((bsz, max_len), dtype=torch.float32)
        mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i, (g, e) in enumerate(zip(panel_gene_ids_list, panel_expr_list)):
            L = len(g)
            if L > 0:
                gene_ids[i, :L] = g
                expr[i, :L] = e
                mask[i, :L] = 1

        dataset_ids = torch.stack(dataset_ids_list, dim=0)
        modality_ids = torch.stack(modality_ids_list, dim=0)
        target_expr = torch.stack(target_expr_list, dim=0)
        target_loss_mask = torch.stack(target_loss_mask_list, dim=0)

        if chosen_panel_sizes and all(x == chosen_panel_sizes[0] for x in chosen_panel_sizes):
            batch_panel_size_value = int(chosen_panel_sizes[0])
        else:
            batch_panel_size_value = -1

        if self.batch_panel_sampling and self.fixed_panel_version is None and self.dataset.fixed_panel_version is None:
            batch_panel_version_value = -1
        elif chosen_versions and all(x == chosen_versions[0] for x in chosen_versions):
            batch_panel_version_value = int(chosen_versions[0])
        else:
            batch_panel_version_value = -1

        return {
            "panel_gene_ids": gene_ids,
            "panel_expr": expr,
            "panel_mask": mask,
            "dataset_ids": dataset_ids,
            "modality_ids": modality_ids,
            "dataset_names": dataset_names,
            "target_expr": target_expr,
            "target_loss_mask": target_loss_mask,
            "batch_panel_size": torch.tensor(int(batch_panel_size_value), dtype=torch.long),
            "item_panel_sizes": torch.tensor(chosen_panel_sizes, dtype=torch.long),
            "batch_panel_version": torch.tensor(int(batch_panel_version_value), dtype=torch.long),
        }


def make_collate_panel_batch(
    dataset: MultiDatasetPanelCompletion,
    batch_panel_sampling: bool = True,
    fixed_panel_size: Optional[int] = None,
    fixed_panel_version: Optional[int] = None,
    seed: int = 42,
    min_masked_genes: int = 8,
) -> PanelBatchCollator:
    return PanelBatchCollator(
        dataset=dataset,
        batch_panel_sampling=batch_panel_sampling,
        fixed_panel_size=fixed_panel_size,
        fixed_panel_version=fixed_panel_version,
        seed=seed,
        min_masked_genes=min_masked_genes,
    )


def collate_panel_batch(batch):
    raise RuntimeError(
        "collate_panel_batch now requires dataset-aware construction. "
        "Use make_collate_panel_batch(dataset, ...)."
    )
