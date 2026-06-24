#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

from dataset import (
    FeatureDataset,
    build_eval_loader,
    denormalize_numpy_targets,
    denormalize_tensor_targets,
    prefilter_missing_features,
)
from model import BiomarkerTokenDecoder


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, ensure_ascii=False)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n < 2:
        return np.nan
    concordant = 0.0
    ties = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            diff_true = y_true[i] - y_true[j]
            diff_pred = y_pred[i] - y_pred[j]
            total += 1.0
            if diff_pred == 0:
                ties += 1.0
            elif diff_true * diff_pred > 0:
                concordant += 1.0
    if total == 0:
        return np.nan
    return float((concordant + 0.5 * ties) / total)


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    metric_cols = ckpt["metric_cols"]
    target_stats = ckpt["target_stats"]
    model_args = ckpt["model_args"]
    model = BiomarkerTokenDecoder(
        in_dim=model_args["in_dim"],
        num_metrics=model_args["num_metrics"],
        d_model=model_args["d_model"],
        num_context_tokens=model_args["num_context_tokens"],
        num_heads=model_args["num_heads"],
        dropout=model_args["dropout"],
        self_attn_layers=model_args["self_attn_layers"],
        cross_attn_layers=model_args["cross_attn_layers"],
        ff_mult=model_args.get("ff_mult", 4),
        spatial_dim=model_args.get("spatial_dim", 3),
        use_feature_layernorm=model_args.get("use_feature_layernorm", True),
        use_feature_l2_norm=model_args.get("use_feature_l2_norm", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, metric_cols, target_stats


def find_checkpoint_for_fold(ckpt_dir: Path, fold_id: int) -> Path:
    candidates = [
        ckpt_dir / f"fold_{fold_id}" / "checkpoints" / "best.pt",
        ckpt_dir / f"fold{fold_id}" / "checkpoints" / "best.pt",
        ckpt_dir / f"fold_{fold_id}" / f"fold_{fold_id}" / "checkpoints" / "best.pt",
        ckpt_dir / f"fold{fold_id}" / f"fold{fold_id}" / "checkpoints" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find best.pt for fold {fold_id}")


@torch.no_grad()
def evaluate_ensemble(models, fold_target_stats, loader, device, metric_cols: List[str], dataset_target_stats):
    metric_abs_sum = {metric_name: 0.0 for metric_name in metric_cols}
    metric_sq_sum = {metric_name: 0.0 for metric_name in metric_cols}
    metric_count = {metric_name: 0 for metric_name in metric_cols}
    pred_collect = {metric_name: [] for metric_name in metric_cols}
    true_collect = {metric_name: [] for metric_name in metric_cols}
    prediction_rows = []

    for batch in loader:
        x, y_norm, mask, slide_ids, coord_strs, coords_xy, spatial_feats = batch
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        spatial_feats = spatial_feats.to(device, non_blocking=True)

        pred_per_fold = []
        for model, target_stats in zip(models, fold_target_stats):
            pred_norm = model(x, spatial_feats=spatial_feats)
            pred_raw = denormalize_tensor_targets(pred_norm, metric_cols, target_stats)
            pred_per_fold.append(pred_raw)
        pred_mean = torch.stack(pred_per_fold, dim=0).mean(dim=0)
        y_true = denormalize_tensor_targets(y_norm, metric_cols, dataset_target_stats)

        pred_np = pred_mean.cpu().numpy()
        y_np = y_true.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)
        abs_err = np.abs(pred_np - y_np)
        sq_err = (pred_np - y_np) ** 2

        for batch_index, slide_id in enumerate(slide_ids):
            for metric_index, metric_name in enumerate(metric_cols):
                if not mask_np[batch_index, metric_index]:
                    continue
                prediction_rows.append(
                    {
                        "slide_id": slide_id,
                        "coord": coord_strs[batch_index],
                        "metric": metric_name,
                        "pred": float(pred_np[batch_index, metric_index]),
                        "target": float(y_np[batch_index, metric_index]),
                    }
                )

        for metric_index, metric_name in enumerate(metric_cols):
            valid = mask_np[:, metric_index]
            if valid.sum() == 0:
                continue
            metric_abs_sum[metric_name] += float(abs_err[valid, metric_index].sum())
            metric_sq_sum[metric_name] += float(sq_err[valid, metric_index].sum())
            metric_count[metric_name] += int(valid.sum())
            pred_collect[metric_name].extend(pred_np[valid, metric_index].tolist())
            true_collect[metric_name].extend(y_np[valid, metric_index].tolist())

    mae_list = []
    rmse_list = []
    pearson_list = []
    spearman_list = []
    cindex_list = []
    per_metric = {}
    for metric_name in metric_cols:
        if metric_count[metric_name] == 0:
            continue
        mae = metric_abs_sum[metric_name] / metric_count[metric_name]
        mse = metric_sq_sum[metric_name] / metric_count[metric_name]
        rmse = math.sqrt(mse)
        y_true = np.asarray(true_collect[metric_name], dtype=float)
        y_pred = np.asarray(pred_collect[metric_name], dtype=float)
        if len(y_true) >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
            pearson_value = pearsonr(y_true, y_pred)[0]
            spearman_value = spearmanr(y_true, y_pred).correlation
        else:
            pearson_value = np.nan
            spearman_value = np.nan
        cindex_value = concordance_index(y_true, y_pred)
        mae_list.append(mae)
        rmse_list.append(rmse)
        if not np.isnan(pearson_value):
            pearson_list.append(float(pearson_value))
        if not np.isnan(spearman_value):
            spearman_list.append(float(spearman_value))
        if not np.isnan(cindex_value):
            cindex_list.append(float(cindex_value))
        per_metric[metric_name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson": float(pearson_value) if not np.isnan(pearson_value) else None,
            "spearman": float(spearman_value) if not np.isnan(spearman_value) else None,
            "c_index": float(cindex_value) if not np.isnan(cindex_value) else None,
            "n": int(metric_count[metric_name]),
        }

    return {
        "mae_mean": float(np.mean(mae_list)) if mae_list else float("nan"),
        "rmse_mean": float(np.mean(rmse_list)) if rmse_list else float("nan"),
        "pearson_mean": float(np.mean(pearson_list)) if pearson_list else float("nan"),
        "spearman_mean": float(np.mean(spearman_list)) if spearman_list else float("nan"),
        "c_index_mean": float(np.mean(cindex_list)) if cindex_list else float("nan"),
        "per_metric": per_metric,
    }, prediction_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    num_folds = int(args.num_folds if args.num_folds is not None else cfg["split"].get("num_folds", 5))
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    pqt_path = Path(cfg["data"]["pqt_path"])
    features_dir = Path(cfg["data"]["features_dir"])
    output_dir = Path(cfg["test"]["output_dir"])
    pred_path = output_dir / "test_predictions.csv"
    ensure_dir(output_dir)

    df = pd.read_parquet(pqt_path)
    if cfg["data"].get("slide_id_to_feature_dir") is not None:
        mapping = cfg["data"]["slide_id_to_feature_dir"]
        df["slide_id"] = df["slide_id"].astype(str)
        df["feature_slide_id"] = df["slide_id"].map(mapping).fillna(df["slide_id"])
    elif "feature_slide_id" not in df.columns:
        df["feature_slide_id"] = df["slide_id"].astype(str)

    test_slides = set(cfg["split"]["test_slides"])
    df_test = prefilter_missing_features(df[df["slide_id"].isin(test_slides)].copy(), features_dir)

    models = []
    fold_target_stats = []
    metric_cols_ref = None
    dataset_target_stats = None
    for fold_id in range(num_folds):
        ckpt_path = find_checkpoint_for_fold(ckpt_dir, fold_id)
        model, metric_cols, target_stats = load_model_from_ckpt(ckpt_path, device)
        models.append(model)
        fold_target_stats.append(target_stats)
        if metric_cols_ref is None:
            metric_cols_ref = metric_cols
            dataset_target_stats = target_stats
        elif list(metric_cols_ref) != list(metric_cols):
            raise ValueError("metric_cols mismatch across folds")

    test_dataset = FeatureDataset(
        df=df_test,
        features_dir=features_dir,
        metric_cols=metric_cols_ref,
        target_stats=dataset_target_stats,
        spatial_dim=int(cfg["model"].get("spatial_dim", 3)),
    )
    test_loader = build_eval_loader(
        dataset=test_dataset,
        batch_size=int(cfg["test"]["batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )
    metrics, prediction_rows = evaluate_ensemble(
        models=models,
        fold_target_stats=fold_target_stats,
        loader=test_loader,
        device=device,
        metric_cols=metric_cols_ref,
        dataset_target_stats=dataset_target_stats,
    )
    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)
    save_json(metrics, output_dir / "test_metrics.json")
    pd.DataFrame(metrics["per_metric"]).T.to_csv(output_dir / "test_metrics_per_metric.csv")
    print("[Test] metrics saved to", output_dir)


if __name__ == "__main__":
    main()
