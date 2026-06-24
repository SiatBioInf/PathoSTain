#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import (
    build_datasets_from_config,
    build_eval_loader,
    build_slide_aware_train_loader,
    compute_target_stats,
    denormalize_tensor_targets,
    list_metric_columns,
    prefilter_missing_features,
)
from losses import local_rank_loss, masked_mse_loss
from model import BiomarkerTokenDecoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, ensure_ascii=False)


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


def resolve_train_val_test_slides(cfg: dict):
    split_cfg = cfg["split"]
    test_slides = set(split_cfg.get("test_slides", []))
    if "train_slides" in split_cfg and "val_slides" in split_cfg:
        return set(split_cfg["train_slides"]), set(split_cfg["val_slides"]), test_slides
    if not bool(split_cfg.get("use_kfold", False)):
        raise KeyError("split.train_slides / split.val_slides not found and split.use_kfold is false.")
    candidate_slides = [str(x) for x in split_cfg.get("candidate_slides", []) if x not in test_slides]
    if len(candidate_slides) == 0:
        raise ValueError("split.candidate_slides is empty.")
    num_folds = int(split_cfg.get("num_folds", 5))
    fold_id = int(split_cfg.get("fold_id", 0))
    if not (0 <= fold_id < num_folds):
        raise ValueError(f"split.fold_id must be in [0, {num_folds - 1}]")
    if bool(split_cfg.get("shuffle", True)):
        rng = np.random.RandomState(int(split_cfg.get("random_seed", 42)))
        rng.shuffle(candidate_slides)
    folds = [list(arr) for arr in np.array_split(np.asarray(candidate_slides, dtype=object), num_folds)]
    val_slides = set(str(x) for x in folds[fold_id])
    train_slides = set()
    for index, fold in enumerate(folds):
        if index == fold_id:
            continue
        train_slides.update(str(x) for x in fold)
    return train_slides, val_slides, test_slides


def build_selection_score(metrics: Dict[str, float], cfg: dict) -> float:
    selection_metric = str(cfg["train"].get("selection_metric", "ranking_score"))
    if selection_metric != "ranking_score":
        value = metrics.get(selection_metric, float("nan"))
        return float(value)

    weights = cfg["train"].get("ranking_score_weights", {"spearman": 0.5, "c_index": 0.5})
    score = 0.0
    denom = 0.0
    for metric_name, weight in weights.items():
        metric_key = f"{metric_name}_mean"
        metric_value = metrics.get(metric_key, float("nan"))
        if np.isnan(metric_value):
            continue
        score += float(weight) * float(metric_value)
        denom += float(weight)
    if denom == 0:
        return float("nan")
    return score / denom


def metric_is_higher_better(cfg: dict) -> bool:
    selection_metric = str(cfg["train"].get("selection_metric", "ranking_score"))
    return selection_metric not in {"mae_mean", "rmse_mean"}


@torch.no_grad()
def evaluate_regression(
    model,
    loader,
    device,
    metric_cols: List[str],
    target_stats: Dict[str, Dict[str, float]],
    cfg: dict,
):
    model.eval()
    metric_abs_sum = {metric_name: 0.0 for metric_name in metric_cols}
    metric_sq_sum = {metric_name: 0.0 for metric_name in metric_cols}
    metric_count = {metric_name: 0 for metric_name in metric_cols}
    pred_collect = {metric_name: [] for metric_name in metric_cols}
    true_collect = {metric_name: [] for metric_name in metric_cols}

    for batch in loader:
        x, y_norm, mask, slide_ids, coord_strs, coords_xy, spatial_feats = batch
        x = x.to(device, non_blocking=True)
        y_norm = y_norm.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        spatial_feats = spatial_feats.to(device, non_blocking=True)

        pred_norm = model(x, spatial_feats=spatial_feats)
        pred = denormalize_tensor_targets(pred_norm, metric_cols, target_stats)
        y_true = denormalize_tensor_targets(y_norm, metric_cols, target_stats)
        abs_err = torch.abs(pred - y_true)
        sq_err = (pred - y_true) ** 2
        pred_np = pred.cpu().numpy()
        y_np = y_true.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        for metric_index, metric_name in enumerate(metric_cols):
            valid = mask_np[:, metric_index]
            if valid.sum() == 0:
                continue
            valid_t = mask[:, metric_index] > 0.5
            metric_abs_sum[metric_name] += float(abs_err[:, metric_index][valid_t].sum().item())
            metric_sq_sum[metric_name] += float(sq_err[:, metric_index][valid_t].sum().item())
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

    results = {
        "mae_mean": float(np.mean(mae_list)) if mae_list else float("nan"),
        "rmse_mean": float(np.mean(rmse_list)) if rmse_list else float("nan"),
        "pearson_mean": float(np.mean(pearson_list)) if pearson_list else float("nan"),
        "spearman_mean": float(np.mean(spearman_list)) if spearman_list else float("nan"),
        "c_index_mean": float(np.mean(cindex_list)) if cindex_list else float("nan"),
        "per_metric": per_metric,
    }
    results["selection_score"] = build_selection_score(results, cfg=cfg)
    return results


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_val_score: float,
    metric_cols: List[str],
    target_stats: Dict[str, Dict[str, float]],
    cfg: dict,
):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_score": best_val_score,
        "metric_cols": metric_cols,
        "target_stats": target_stats,
        "model_args": {
            "in_dim": cfg["model"]["in_dim"],
            "num_metrics": len(metric_cols),
            "d_model": cfg["model"]["d_model"],
            "num_context_tokens": cfg["model"]["num_context_tokens"],
            "num_heads": cfg["model"]["num_heads"],
            "dropout": cfg["model"]["dropout"],
            "self_attn_layers": cfg["model"]["self_attn_layers"],
            "cross_attn_layers": cfg["model"]["cross_attn_layers"],
            "ff_mult": cfg["model"].get("ff_mult", 4),
            "spatial_dim": cfg["model"].get("spatial_dim", 3),
            "use_feature_layernorm": cfg["model"].get("use_feature_layernorm", True),
            "use_feature_l2_norm": cfg["model"].get("use_feature_l2_norm", True),
        },
        "config": cfg,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    pqt_path = Path(cfg["data"]["pqt_path"])
    features_dir = Path(cfg["data"]["features_dir"])
    output_dir = Path(cfg["train"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    pred_dir = output_dir / "val_predictions"
    for folder in [output_dir, ckpt_dir, log_dir, pred_dir]:
        ensure_dir(folder)

    print("[Info] device       =", device)
    print("[Info] pqt_path     =", pqt_path)
    print("[Info] features_dir =", features_dir)
    print("[Info] output_dir   =", output_dir)

    df = pd.read_parquet(pqt_path)
    for column_name in ["slide_id", "coord", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
        if column_name not in df.columns:
            raise KeyError(f"Missing column in parquet: {column_name}")

    if cfg["data"].get("slide_id_to_feature_dir") is not None:
        mapping = cfg["data"]["slide_id_to_feature_dir"]
        df["slide_id"] = df["slide_id"].astype(str)
        df["feature_slide_id"] = df["slide_id"].map(mapping).fillna(df["slide_id"])
    elif "feature_slide_id" not in df.columns:
        df["feature_slide_id"] = df["slide_id"].astype(str)

    metric_cols = list_metric_columns(df)
    print("[Info] num_metrics =", len(metric_cols))
    train_slides, val_slides, test_slides = resolve_train_val_test_slides(cfg)
    print("[Info] train_slides =", sorted(list(train_slides)))
    print("[Info] val_slides   =", sorted(list(val_slides)))
    print("[Info] test_slides  =", sorted(list(test_slides)))

    overlap = (train_slides & val_slides) | (train_slides & test_slides) | (val_slides & test_slides)
    if overlap:
        raise ValueError(f"Overlap detected among train/val/test slides: {sorted(list(overlap))}")

    df_train = prefilter_missing_features(df[df["slide_id"].isin(train_slides)].copy(), features_dir)
    df_val = prefilter_missing_features(df[df["slide_id"].isin(val_slides)].copy(), features_dir)
    print("[Info] after prefilter: train =", len(df_train), "val =", len(df_val))

    target_stats = compute_target_stats(df_train, metric_cols, cfg)
    save_json({"metric_cols": metric_cols, "target_stats": target_stats}, output_dir / "target_stats.json")

    train_dataset, val_dataset = build_datasets_from_config(
        df_train=df_train,
        df_val=df_val,
        features_dir=features_dir,
        metric_cols=metric_cols,
        target_stats=target_stats,
        cfg=cfg,
    )
    train_loader = build_slide_aware_train_loader(
        dataset=train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        spots_per_slide=int(cfg["train"]["spots_per_slide"]),
        slides_per_batch=int(cfg["train"]["slides_per_batch"]),
        num_workers=int(cfg["train"]["num_workers"]),
        seed=seed,
    )
    val_loader = build_eval_loader(
        dataset=val_dataset,
        batch_size=int(cfg["train"]["eval_batch_size"]),
        num_workers=int(cfg["train"]["num_workers"]),
    )

    model = BiomarkerTokenDecoder(
        in_dim=int(cfg["model"]["in_dim"]),
        num_metrics=len(metric_cols),
        d_model=int(cfg["model"]["d_model"]),
        num_context_tokens=int(cfg["model"]["num_context_tokens"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        self_attn_layers=int(cfg["model"]["self_attn_layers"]),
        cross_attn_layers=int(cfg["model"]["cross_attn_layers"]),
        ff_mult=int(cfg["model"].get("ff_mult", 4)),
        spatial_dim=int(cfg["model"].get("spatial_dim", 3)),
        use_feature_layernorm=bool(cfg["model"].get("use_feature_layernorm", True)),
        use_feature_l2_norm=bool(cfg["model"].get("use_feature_l2_norm", True)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if metric_is_higher_better(cfg) else "min",
        factor=float(cfg["optim"]["lr_decay_factor"]),
        patience=int(cfg["optim"]["lr_patience"]),
        min_lr=float(cfg["optim"]["min_lr"]),
    )

    max_epochs = int(cfg["train"]["max_epochs"])
    min_epochs = int(cfg["train"]["min_epochs"])
    early_patience = int(cfg["train"]["early_stop_patience"])
    grad_clip = float(cfg["train"]["grad_clip"])
    lambda_mse = float(cfg["loss"]["lambda_mse"])
    lambda_local_rank = float(cfg["loss"]["lambda_local_rank"])
    local_rank_radius = float(cfg["loss"]["local_rank_radius_px"])
    local_rank_margin = float(cfg["loss"]["local_rank_margin"])
    local_rank_max_pairs = int(cfg["loss"]["local_rank_max_pairs_per_metric"])

    better_is_higher = metric_is_higher_better(cfg)
    best_val_score = -float("inf") if better_is_higher else float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    print("[Train] start")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        train_rank_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            x, y_norm, mask, slide_ids, coord_strs, coords_xy, spatial_feats = batch
            x = x.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            coords_xy = coords_xy.to(device, non_blocking=True)
            spatial_feats = spatial_feats.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x, spatial_feats=spatial_feats)
            loss_mse = masked_mse_loss(pred, y_norm, mask)
            loss_rank = local_rank_loss(
                pred=pred,
                target=y_norm,
                mask=mask,
                slide_ids=slide_ids,
                coords_xy=coords_xy,
                radius_px=local_rank_radius,
                margin=local_rank_margin,
                max_pairs_per_metric=local_rank_max_pairs,
                use_distance_weight=bool(cfg["loss"].get("use_distance_weight", False)),
                distance_sigma=float(cfg["loss"].get("distance_sigma", 150.0)),
            )
            loss = lambda_mse * loss_mse + lambda_local_rank * loss_rank
            if torch.isnan(loss):
                print("[Warn] NaN loss encountered, skip batch.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_mse_sum += float(loss_mse.item())
            train_rank_sum += float(loss_rank.item())
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        train_mse = train_mse_sum / max(n_batches, 1)
        train_rank = train_rank_sum / max(n_batches, 1)
        val_metrics = evaluate_regression(model, val_loader, device, metric_cols, target_stats, cfg)
        val_score = float(val_metrics["selection_score"])
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_mse_loss": train_mse,
            "train_local_rank_loss": train_rank,
            "val_selection_score": val_score,
            "val_mae_mean": val_metrics["mae_mean"],
            "val_rmse_mean": val_metrics["rmse_mean"],
            "val_pearson_mean": val_metrics["pearson_mean"],
            "val_spearman_mean": val_metrics["spearman_mean"],
            "val_c_index_mean": val_metrics["c_index_mean"],
            "lr": current_lr,
        }
        history.append(history_row)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_loss:.4f} | "
            f"train_mse={train_mse:.4f} | "
            f"train_local_rank={train_rank:.4f} | "
            f"val_score={val_score:.4f} | "
            f"val_mae={val_metrics['mae_mean']:.4f} | "
            f"val_spearman={val_metrics['spearman_mean']:.4f} | "
            f"val_cindex={val_metrics['c_index_mean']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        save_checkpoint(
            ckpt_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_score,
            metric_cols,
            target_stats,
            cfg,
        )

        improved = val_score > best_val_score if better_is_higher else val_score < best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_score,
                metric_cols,
                target_stats,
                cfg,
            )
        else:
            epochs_no_improve += 1

        pd.DataFrame(history).to_csv(log_dir / "train_history.csv", index=False)
        if epoch >= min_epochs and epochs_no_improve >= early_patience:
            print(f"[Early Stop] no improvement for {early_patience} epochs after min_epochs={min_epochs}.")
            break

    elapsed = time.time() - start_time
    print("[Done] best_epoch =", best_epoch)
    print("[Done] best_val_score =", best_val_score)
    print("[Done] elapsed_sec =", elapsed)


if __name__ == "__main__":
    main()
