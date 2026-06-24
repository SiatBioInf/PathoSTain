#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


NON_METRIC_COLUMNS = {
    "slide_id",
    "feature_slide_id",
    "coord",
    "barcode",
    "spot_barcode",
    "split",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
    "array_row",
    "array_col",
    "in_tissue",
}


def list_metric_columns(df: pd.DataFrame) -> List[str]:
    metric_cols = []
    for col in df.columns:
        if col in NON_METRIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)
    return metric_cols


def feature_path_of_row(row: pd.Series, features_dir: Path) -> Path:
    slide_folder = str(row["feature_slide_id"])
    coord = str(row["coord"]).strip()
    return features_dir / slide_folder / f"{coord}.pt"


def prefilter_missing_features(df: pd.DataFrame, features_dir: Path) -> pd.DataFrame:
    keep = []
    for index in range(len(df)):
        row = df.iloc[index]
        keep.append(feature_path_of_row(row, features_dir).exists())
    return df.loc[np.asarray(keep, dtype=bool)].copy().reset_index(drop=True)


def _transform_name_for_metric(metric_name: str, cfg: dict) -> str:
    transform_cfg = cfg.get("target_transforms", {})
    moderate = set(transform_cfg.get("moderate_asinh_metrics", []))
    high_zero = set(transform_cfg.get("high_zero_asinh_metrics", []))
    if metric_name in high_zero or metric_name in moderate:
        return "asinh"
    return "identity"


def apply_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return values
    if transform_name == "asinh":
        return np.arcsinh(values)
    raise ValueError(f"Unsupported transform: {transform_name}")


def invert_target_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return values
    if transform_name == "asinh":
        return np.sinh(values)
    raise ValueError(f"Unsupported transform: {transform_name}")


def compute_target_stats(df_train: pd.DataFrame, metric_cols: List[str], cfg: dict) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_cols:
        raw_values = df_train[metric_name].to_numpy(dtype=float)
        raw_values = raw_values[~np.isnan(raw_values)]
        transform_name = _transform_name_for_metric(metric_name, cfg)
        if raw_values.size == 0:
            stats[metric_name] = {"transform": transform_name, "mean": 0.0, "std": 1.0}
            continue
        transformed = apply_target_transform(raw_values, transform_name)
        mean_value = float(np.mean(transformed))
        std_value = float(np.std(transformed))
        if std_value < 1e-6:
            std_value = 1.0
        stats[metric_name] = {
            "transform": transform_name,
            "mean": mean_value,
            "std": std_value,
        }
    return stats


def normalize_target_value(value: float, metric_stat: Dict[str, float]) -> float:
    transformed = apply_target_transform(np.asarray([value], dtype=np.float64), metric_stat["transform"])[0]
    return float((transformed - float(metric_stat["mean"])) / float(metric_stat["std"]))


def denormalize_tensor_targets(
    values: torch.Tensor,
    metric_cols: List[str],
    target_stats: Dict[str, Dict[str, float]],
) -> torch.Tensor:
    out = values.clone()
    device = out.device
    dtype = out.dtype
    for metric_index, metric_name in enumerate(metric_cols):
        metric_stat = target_stats[metric_name]
        transformed = out[:, metric_index] * float(metric_stat["std"]) + float(metric_stat["mean"])
        if metric_stat["transform"] == "identity":
            raw_value = transformed
        elif metric_stat["transform"] == "asinh":
            raw_value = torch.sinh(transformed)
        else:
            raise ValueError(f"Unsupported transform: {metric_stat['transform']}")
        out[:, metric_index] = raw_value.to(device=device, dtype=dtype)
    return out


def denormalize_numpy_targets(
    values: np.ndarray,
    metric_cols: List[str],
    target_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for metric_index, metric_name in enumerate(metric_cols):
        metric_stat = target_stats[metric_name]
        transformed = out[:, metric_index] * float(metric_stat["std"]) + float(metric_stat["mean"])
        out[:, metric_index] = invert_target_transform(transformed, metric_stat["transform"])
    return out


class FeatureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features_dir: Path,
        metric_cols: List[str],
        target_stats: Dict[str, Dict[str, float]],
        spatial_dim: int = 3,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.features_dir = Path(features_dir)
        self.metric_cols = metric_cols
        self.target_stats = target_stats
        self.spatial_dim = int(spatial_dim)

        self.slide_to_indices: Dict[str, List[int]] = {}
        self.slide_coord_stats: Dict[str, Dict[str, float]] = {}
        for index in range(len(self.df)):
            slide_id = str(self.df.iloc[index]["slide_id"])
            self.slide_to_indices.setdefault(slide_id, []).append(index)

        for slide_id, indices in self.slide_to_indices.items():
            coords = self.df.loc[indices, ["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(dtype=float)
            row_mean = float(coords[:, 0].mean())
            col_mean = float(coords[:, 1].mean())
            row_std = float(coords[:, 0].std()) if float(coords[:, 0].std()) > 1e-6 else 1.0
            col_std = float(coords[:, 1].std()) if float(coords[:, 1].std()) > 1e-6 else 1.0
            self.slide_coord_stats[slide_id] = {
                "row_mean": row_mean,
                "row_std": row_std,
                "col_mean": col_mean,
                "col_std": col_std,
            }

        self.unique_slides = sorted(self.slide_to_indices.keys())

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_feature(feat_path: Path) -> torch.Tensor:
        obj = torch.load(feat_path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            feat = obj.float()
        elif isinstance(obj, np.ndarray):
            feat = torch.from_numpy(obj).float()
        elif isinstance(obj, dict):
            if "feature" in obj:
                feat = obj["feature"]
            elif "feat" in obj:
                feat = obj["feat"]
            elif "x" in obj:
                feat = obj["x"]
            else:
                raise KeyError(f"Unsupported dict keys in feature file: {feat_path}")
            feat = torch.as_tensor(feat, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported feature object type: {type(obj)} in {feat_path}")
        return feat.view(-1).float()

    def _build_spatial_feature(self, row: pd.Series) -> torch.Tensor:
        slide_id = str(row["slide_id"])
        coord_stats = self.slide_coord_stats[slide_id]
        row_z = (float(row["pxl_row_in_fullres"]) - coord_stats["row_mean"]) / coord_stats["row_std"]
        col_z = (float(row["pxl_col_in_fullres"]) - coord_stats["col_mean"]) / coord_stats["col_std"]
        radius = math.sqrt(row_z * row_z + col_z * col_z)
        spatial = [row_z, col_z, radius]
        return torch.tensor(spatial[: self.spatial_dim], dtype=torch.float32)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        feature_path = feature_path_of_row(row, self.features_dir)
        x = self._load_feature(feature_path)

        y_values = []
        mask_values = []
        for metric_name in self.metric_cols:
            value = row[metric_name]
            if pd.isna(value):
                y_values.append(0.0)
                mask_values.append(0.0)
            else:
                y_values.append(normalize_target_value(float(value), self.target_stats[metric_name]))
                mask_values.append(1.0)

        y = torch.tensor(y_values, dtype=torch.float32)
        mask = torch.tensor(mask_values, dtype=torch.float32)
        slide_id = str(row["slide_id"])
        coord = str(row["coord"]).strip()
        coords_xy = torch.tensor(
            [float(row["pxl_row_in_fullres"]), float(row["pxl_col_in_fullres"])],
            dtype=torch.float32,
        )
        spatial_feats = self._build_spatial_feature(row)
        return x, y, mask, slide_id, coord, coords_xy, spatial_feats


def feature_collate_fn(batch):
    xs, ys, masks, slide_ids, coord_strs, coords_xy, spatial_feats = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    mask = torch.stack(masks, dim=0)
    coords_xy = torch.stack(coords_xy, dim=0)
    spatial_feats = torch.stack(spatial_feats, dim=0)
    return x, y, mask, list(slide_ids), list(coord_strs), coords_xy, spatial_feats


class SlideAwareBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: FeatureDataset,
        slides_per_batch: int,
        spots_per_slide: int,
        seed: int = 42,
        batches_per_epoch: Optional[int] = None,
    ):
        self.dataset = dataset
        self.slides_per_batch = int(slides_per_batch)
        self.spots_per_slide = int(spots_per_slide)
        self.seed = int(seed)
        self.unique_slides = dataset.unique_slides
        self.slide_to_indices = dataset.slide_to_indices
        if len(self.unique_slides) == 0:
            raise ValueError("No slides found in dataset.")
        if batches_per_epoch is None:
            eff_batch = self.slides_per_batch * self.spots_per_slide
            batches_per_epoch = max(1, math.ceil(len(dataset) / max(eff_batch, 1)))
        self.batches_per_epoch = int(batches_per_epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + random.randint(0, 10_000_000))
        for _ in range(self.batches_per_epoch):
            chosen_slides = rng.sample(self.unique_slides, k=min(self.slides_per_batch, len(self.unique_slides)))
            batch_indices: List[int] = []
            for slide_id in chosen_slides:
                candidates = self.slide_to_indices[slide_id]
                if len(candidates) >= self.spots_per_slide:
                    picked = rng.sample(candidates, k=self.spots_per_slide)
                else:
                    picked = [rng.choice(candidates) for _ in range(self.spots_per_slide)]
                batch_indices.extend(picked)
            rng.shuffle(batch_indices)
            yield batch_indices


def build_datasets_from_config(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    features_dir: Path,
    metric_cols: List[str],
    target_stats: Dict[str, Dict[str, float]],
    cfg: dict,
):
    spatial_dim = int(cfg["model"].get("spatial_dim", 3))
    train_dataset = FeatureDataset(
        df=df_train,
        features_dir=features_dir,
        metric_cols=metric_cols,
        target_stats=target_stats,
        spatial_dim=spatial_dim,
    )
    val_dataset = FeatureDataset(
        df=df_val,
        features_dir=features_dir,
        metric_cols=metric_cols,
        target_stats=target_stats,
        spatial_dim=spatial_dim,
    )
    return train_dataset, val_dataset


def build_slide_aware_train_loader(
    dataset: FeatureDataset,
    batch_size: int,
    spots_per_slide: int,
    slides_per_batch: int,
    num_workers: int,
    seed: int,
):
    eff_batch = slides_per_batch * spots_per_slide
    if eff_batch != batch_size:
        print(
            f"[Warn] batch_size={batch_size} != slides_per_batch*spots_per_slide={eff_batch}. "
            f"Actual batch size follows sampler={eff_batch}."
        )
    sampler = SlideAwareBatchSampler(
        dataset=dataset,
        slides_per_batch=slides_per_batch,
        spots_per_slide=spots_per_slide,
        seed=seed,
        batches_per_epoch=None,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=feature_collate_fn,
    )


def build_eval_loader(dataset: FeatureDataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=feature_collate_fn,
    )
