from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from impleo.data import MultiDatasetPanelCompletion, make_collate_panel_batch
from impleo.model import PanelCompletionModel
from impleo.losses import compute_panel_loss, huber_elementwise


DEFAULTS = {
    "batch_size": 64,
    "epochs": 6,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "panel_sizes": [98, 128, 256, 512],
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "num_workers": 4,
    "eval_every": 1,
    "device": "cuda",
    "seed": 42,
    "split_mode": "dataset_holdout",
    "use_dataset_emb": False,
    "use_modality_emb": True,
    "dataset_split_file": None,
    "train_datasets": None,
    "val_datasets": None,
    "num_panel_versions": 5,
    "panel_pool_multiplier": 4.0,
    "panel_anchor": 32,
    "min_masked_genes": 8,
    "decoder_type": "cross_attn",
    "decoder_query_chunk_size": 1024,
    "decoder_use_concat_query": True,
    "loss_type": "huber_corr",
    "huber_delta": 1.0,
    "corr_lambda": 0.05,
    "corr_min_genes": 2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model: PanelCompletionModel, loader: DataLoader, device: torch.device, cfg: Dict[str, object]) -> Dict[str, float]:
    model.eval()

    total_mse = 0.0
    total_huber = 0.0
    total_weight = 0.0
    total_objective = 0.0
    total_corr = 0.0
    steps = 0

    per_dataset_mse_sum: Dict[str, float] = {}
    per_dataset_huber_sum: Dict[str, float] = {}
    per_dataset_weight: Dict[str, float] = {}

    loss_type = str(cfg.get("loss_type", "mse"))
    huber_delta = float(cfg.get("huber_delta", 1.0))
    corr_lambda = float(cfg.get("corr_lambda", 0.05))
    corr_min_genes = int(cfg.get("corr_min_genes", 2))

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        pred = model(
            panel_gene_ids=batch["panel_gene_ids"],
            panel_expr=batch["panel_expr"],
            panel_mask=batch["panel_mask"],
            dataset_ids=batch["dataset_ids"],
            modality_ids=batch["modality_ids"],
        )

        target = batch["target_expr"]
        mask = batch["target_loss_mask"].to(dtype=pred.dtype)

        loss_out = compute_panel_loss(
            pred,
            target,
            mask,
            loss_type=loss_type,
            huber_delta=huber_delta,
            corr_lambda=corr_lambda,
            corr_min_genes=corr_min_genes,
        )

        total_objective += float(loss_out["loss"].detach().item())
        total_corr += float(loss_out["corr_loss"].detach().item())
        steps += 1

        diff2 = ((pred - target) ** 2) * mask
        huber = huber_elementwise(pred - target, delta=huber_delta) * mask

        total_mse += float(diff2.sum().detach().item())
        total_huber += float(huber.sum().detach().item())
        total_weight += float(mask.sum().detach().item())

        row_weight = mask.sum(dim=1)
        valid_rows = row_weight > 0

        row_mse = torch.zeros_like(row_weight, dtype=torch.float32)
        row_huber = torch.zeros_like(row_weight, dtype=torch.float32)

        row_mse[valid_rows] = diff2.sum(dim=1)[valid_rows] / row_weight[valid_rows]
        row_huber[valid_rows] = huber.sum(dim=1)[valid_rows] / row_weight[valid_rows]

        for ds_name, mse, huber_v, ok in zip(
            batch["dataset_names"],
            row_mse.tolist(),
            row_huber.tolist(),
            valid_rows.tolist(),
        ):
            if not ok:
                continue
            key = str(ds_name)
            per_dataset_mse_sum[key] = per_dataset_mse_sum.get(key, 0.0) + float(mse)
            per_dataset_huber_sum[key] = per_dataset_huber_sum.get(key, 0.0) + float(huber_v)
            per_dataset_weight[key] = per_dataset_weight.get(key, 0.0) + 1.0

    metrics = {
        "objective_loss": total_objective / max(steps, 1),
        "masked_mse": total_mse / max(total_weight, 1.0),
        "masked_huber": total_huber / max(total_weight, 1.0),
        "corr_loss": total_corr / max(steps, 1),
    }

    for k in sorted(per_dataset_mse_sum):
        denom = max(per_dataset_weight[k], 1.0)
        metrics[f"{k}/masked_mse"] = per_dataset_mse_sum[k] / denom
        metrics[f"{k}/masked_huber"] = per_dataset_huber_sum[k] / denom

    return metrics


def train_one_epoch(
    model: PanelCompletionModel,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    cfg: Dict[str, object],
) -> Dict[str, float]:
    model.train()

    running = {
        "loss": 0.0,
        "masked_mse": 0.0,
        "masked_huber": 0.0,
        "corr_loss": 0.0,
    }
    steps = 0

    loss_type = str(cfg.get("loss_type", "mse"))
    huber_delta = float(cfg.get("huber_delta", 1.0))
    corr_lambda = float(cfg.get("corr_lambda", 0.05))
    corr_min_genes = int(cfg.get("corr_min_genes", 2))

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        pred = model(
            panel_gene_ids=batch["panel_gene_ids"],
            panel_expr=batch["panel_expr"],
            panel_mask=batch["panel_mask"],
            dataset_ids=batch["dataset_ids"],
            modality_ids=batch["modality_ids"],
        )

        loss_out = compute_panel_loss(
            pred,
            batch["target_expr"],
            batch["target_loss_mask"],
            loss_type=loss_type,
            huber_delta=huber_delta,
            corr_lambda=corr_lambda,
            corr_min_genes=corr_min_genes,
        )

        loss = loss_out["loss"]
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += float(loss_out[k].detach().item())
        steps += 1

    return {
        "train_objective_loss": running["loss"] / max(steps, 1),
        "train_masked_mse": running["masked_mse"] / max(steps, 1),
        "train_masked_huber": running["masked_huber"] / max(steps, 1),
        "train_corr_loss": running["corr_loss"] / max(steps, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train panel-to-transcriptome completion model.")
    parser.add_argument("--output-dir", default=None, help="New checkpoint/log directory")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--panel-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-mode", choices=["within_dataset", "dataset_holdout"], default=None)
    parser.add_argument("--dataset-split-file", type=str, default=None)
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument(
        "--val-datasets", "--test-datasets", dest="val_datasets", nargs="+", default=None,
        help="Validation dataset keys used for checkpoint selection",
    )
    parser.add_argument("--num-panel-versions", type=int, default=None)
    parser.add_argument("--panel-pool-multiplier", type=float, default=None)
    parser.add_argument("--panel-anchor", type=int, default=None)
    parser.add_argument("--min-masked-genes", type=int, default=None)
    parser.add_argument("--decoder-type", choices=["cross_attn"], default=None)
    parser.add_argument("--decoder-query-chunk-size", type=int, default=None)
    parser.add_argument("--loss-type", choices=["mse", "huber", "huber_corr"], default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--corr-lambda", type=float, default=None)
    parser.add_argument("--corr-min-genes", type=int, default=None)
    parser.add_argument("--decoder-use-concat-query", dest="decoder_use_concat_query", action="store_true")
    parser.add_argument("--decoder-no-concat-query", dest="decoder_use_concat_query", action="store_false")
    parser.add_argument("--use-dataset-emb", dest="use_dataset_emb", action="store_true")
    parser.add_argument("--no-dataset-emb", dest="use_dataset_emb", action="store_false")
    parser.add_argument("--use-modality-emb", dest="use_modality_emb", action="store_true")
    parser.add_argument("--no-modality-emb", dest="use_modality_emb", action="store_false")
    parser.set_defaults(
        use_dataset_emb=None,
        use_modality_emb=None,
        decoder_use_concat_query=None,
    )
    return parser.parse_args()


def load_config(args):
    cfg = DEFAULTS.copy()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        if "test_datasets" in yaml_cfg:
            legacy_val = yaml_cfg.pop("test_datasets")
            if "val_datasets" in yaml_cfg and yaml_cfg["val_datasets"] != legacy_val:
                raise ValueError("Conflicting val_datasets and legacy test_datasets")
            yaml_cfg.setdefault("val_datasets", legacy_val)
        cfg.update(yaml_cfg)

    cli = vars(args).copy()
    cli.pop("config", None)
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v

    if cfg.get("dataset_split_file") and args.config:
        split = Path(cfg["dataset_split_file"])
        if not split.is_absolute():
            cfg["dataset_split_file"] = str((Path(args.config).resolve().parent / split).resolve())

    if cfg.get("processed_dir") is None:
        raise ValueError("--processed-dir is required")

    if cfg["split_mode"] == "dataset_holdout" and cfg.get("use_dataset_emb", True):
        print("[warn] split_mode=dataset_holdout with dataset embeddings is not recommended for unseen validation datasets.")
        print("[warn] Automatically disabling dataset embeddings.")
        cfg["use_dataset_emb"] = False

    return cfg


def _read_json_or_yaml(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_dataset_splits(cfg: Dict[str, object]) -> Tuple[List[str], List[str]]:
    train_keys = cfg.get("train_datasets")
    val_keys = cfg.get("val_datasets")

    if cfg.get("dataset_split_file"):
        obj = _read_json_or_yaml(Path(str(cfg["dataset_split_file"])))
        if not isinstance(obj, dict):
            raise ValueError("dataset_split_file must contain a dict with keys 'train' and 'val'")
        train_keys = obj.get("train", train_keys)
        val_keys = obj.get("val", obj.get("test", val_keys))

    if not train_keys and not val_keys and not cfg.get("dataset_split_file"):
        checkpoint = Path(__file__).resolve().parents[1] / "checkpoints" / "impleo.pt"
        article = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
        train_keys = article["train_datasets"]
        val_keys = article.get("val_datasets", article.get("test_datasets"))

    if not train_keys or not val_keys:
        raise ValueError(
            "For split_mode=dataset_holdout, you must provide train/validation dataset keys via "
            "--dataset-split-file or --train-datasets/--val-datasets"
        )

    train_list = [str(x) for x in train_keys]
    val_list = [str(x) for x in val_keys]
    overlap = sorted(set(train_list).intersection(val_list))
    if overlap:
        raise ValueError(f"Datasets appear in both train and validation: {overlap}")
    return train_list, val_list


def build_datasets(cfg: Dict[str, object]) -> Tuple[MultiDatasetPanelCompletion, MultiDatasetPanelCompletion, Dict[str, object]]:
    split_mode = str(cfg.get("split_mode", "within_dataset"))
    processed_dir = str(cfg["processed_dir"])
    panel_sizes = cfg["panel_sizes"]
    seed = int(cfg["seed"])

    common_kwargs = dict(
        panel_sizes=panel_sizes,
        num_panel_versions=int(cfg["num_panel_versions"]),
        panel_pool_multiplier=float(cfg["panel_pool_multiplier"]),
        panel_anchor=int(cfg["panel_anchor"]),
    )

    if split_mode == "within_dataset":
        train_ds = MultiDatasetPanelCompletion(
            processed_dir=processed_dir,
            split="train",
            seed=seed,
            **common_kwargs,
        )
        # Existing within-dataset assets store validation rows under the "test" filename.
        val_ds = MultiDatasetPanelCompletion(
            processed_dir=processed_dir,
            split="test",
            seed=seed + 1,
            dataset_id_map=train_ds.dataset2id,
            modality_id_map=train_ds.modality2id,
            **common_kwargs,
        )
        split_meta = {
            "split_mode": "within_dataset",
            "train_datasets": sorted(train_ds.dataset2id.keys()),
            "val_datasets": sorted(val_ds.dataset2id.keys()),
            "train_matrix_split": "train",
            "val_matrix_split": "test",
        }
        return train_ds, val_ds, split_meta

    train_keys, val_keys = resolve_dataset_splits(cfg)
    train_ds = MultiDatasetPanelCompletion(
        processed_dir=processed_dir,
        split="full",
        seed=seed,
        dataset_keys=train_keys,
        **common_kwargs,
    )
    val_ds = MultiDatasetPanelCompletion(
        processed_dir=processed_dir,
        split="full",
        seed=seed + 1,
        dataset_keys=val_keys,
        modality_id_map=train_ds.modality2id,
        modality_id_fallback=0,
        **common_kwargs,
    )
    split_meta = {
        "split_mode": "dataset_holdout",
        "train_datasets": train_keys,
        "val_datasets": val_keys,
        "train_matrix_split": "full",
        "val_matrix_split": "full",
    }
    return train_ds, val_ds, split_meta


def main():
    args = parse_args()
    cfg = load_config(args)
    set_seed(int(cfg["seed"]))

    device = torch.device(cfg["device"] if (cfg["device"] == "cpu" or torch.cuda.is_available()) else "cpu")

    train_ds, val_ds, split_meta = build_datasets(cfg)
    cfg.update(split_meta)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        collate_fn=make_collate_panel_batch(
            train_ds,
            batch_panel_sampling=True,
            seed=int(cfg["seed"]),
            min_masked_genes=int(cfg["min_masked_genes"]),
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        collate_fn=make_collate_panel_batch(
            val_ds,
            batch_panel_sampling=True,
            seed=int(cfg["seed"]) + 1000,
            min_masked_genes=int(cfg["min_masked_genes"]),
        ),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = PanelCompletionModel(
        num_input_genes=len(train_ds.vocab),
        num_output_genes=len(train_ds.vocab),
        num_datasets=max(1, len(train_ds.dataset2id)),
        num_modalities=max(1, len(train_ds.modality2id)),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
        use_dataset_emb=bool(cfg.get("use_dataset_emb", True)),
        use_modality_emb=bool(cfg.get("use_modality_emb", True)),
        decoder_type=str(cfg.get("decoder_type", "baseline")),
        decoder_query_chunk_size=int(cfg.get("decoder_query_chunk_size", 1024)),
        decoder_use_concat_query=bool(cfg.get("decoder_use_concat_query", True)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    run_dir = Path(cfg.get("output_dir") or (Path("results/pretraining") / time.strftime("%Y%m%d_%H%M%S")))
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    best = math.inf
    history: List[Dict[str, float]] = []
    runtime_rows: List[Dict[str, float]] = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg)
        train_sec = time.perf_counter() - t0

        record: Dict[str, float] = {"epoch": float(epoch), **{k: float(v) for k, v in train_metrics.items()}, "train_sec": float(train_sec)}

        if epoch % int(cfg["eval_every"]) == 0:
            t1 = time.perf_counter()
            metrics = evaluate(model, val_loader, device, cfg)
            eval_sec = time.perf_counter() - t1
            record["eval_sec"] = float(eval_sec)
            record.update({f"val_{k}": float(v) for k, v in metrics.items()})

            cur = float(metrics["objective_loss"])
            if cur < best:
                best = cur
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "vocab": train_ds.vocab,
                        "dataset2id": train_ds.dataset2id,
                        "modality2id": train_ds.modality2id,
                    },
                    run_dir / "best.pt",
                )

        peak_mem_gb = 0.0
        if device.type == "cuda":
            peak_mem_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
        record["peak_cuda_mem_gb"] = peak_mem_gb

        history.append(record)
        runtime_rows.append(
            {
                "epoch": int(epoch),
                "decoder_type": str(cfg.get("decoder_type", "baseline")),
                "split_mode": str(cfg.get("split_mode", "within_dataset")),
                "panel_sizes": ",".join(str(x) for x in cfg["panel_sizes"]),
                "seed": int(cfg["seed"]),
                "train_sec": float(record.get("train_sec", 0.0)),
                "eval_sec": float(record.get("eval_sec", 0.0)),
                "peak_cuda_mem_gb": float(record.get("peak_cuda_mem_gb", 0.0)),
            }
        )
        print(json.dumps(record, ensure_ascii=False))

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "vocab": train_ds.vocab,
                "dataset2id": train_ds.dataset2id,
                "modality2id": train_ds.modality2id,
            },
            run_dir / "last.pt",
        )

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(run_dir / "runtime.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(runtime_rows[0].keys()))
        writer.writeheader()
        writer.writerows(runtime_rows)


if __name__ == "__main__":
    main()
