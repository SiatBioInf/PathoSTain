#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import torch


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(eps)
    return diff2.sum() / denom


def _pairwise_hinge_rank_loss(
    pred_i: torch.Tensor,
    pred_j: torch.Tensor,
    true_i: torch.Tensor,
    true_j: torch.Tensor,
    margin: float = 0.0,
):
    sign = torch.sign(true_i - true_j)
    valid = sign != 0
    if valid.sum() == 0:
        return None
    pred_diff = pred_i - pred_j
    return torch.relu(margin - sign[valid] * pred_diff[valid])


def local_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    slide_ids: List[str],
    coords_xy: torch.Tensor,
    radius_px: float = 300.0,
    margin: float = 0.0,
    max_pairs_per_metric: int = 256,
    use_distance_weight: bool = False,
    distance_sigma: float = 150.0,
    eps: float = 1e-8,
):
    device = pred.device
    batch_size, num_metrics = pred.shape
    if batch_size <= 1:
        return pred.new_tensor(0.0)

    slide_to_indices = {}
    for index, slide_id in enumerate(slide_ids):
        slide_to_indices.setdefault(str(slide_id), []).append(index)

    total_loss = pred.new_tensor(0.0)
    total_terms = 0
    radius2 = float(radius_px) * float(radius_px)

    for _, slide_indices in slide_to_indices.items():
        if len(slide_indices) < 2:
            continue
        slide_index_tensor = torch.as_tensor(slide_indices, dtype=torch.long, device=device)
        coords = coords_xy[slide_index_tensor]
        pred_slide = pred[slide_index_tensor]
        target_slide = target[slide_index_tensor]
        mask_slide = mask[slide_index_tensor]

        num_slide_spots = coords.shape[0]
        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = (diff ** 2).sum(dim=-1)
        tri_mask = torch.triu(torch.ones((num_slide_spots, num_slide_spots), dtype=torch.bool, device=device), diagonal=1)
        pair_mask = tri_mask & (dist2 <= radius2)
        pair_indices = pair_mask.nonzero(as_tuple=False)
        if pair_indices.numel() == 0:
            continue

        for metric_index in range(num_metrics):
            valid_metric = mask_slide[:, metric_index] > 0.5
            if valid_metric.sum() < 2:
                continue
            i_idx = pair_indices[:, 0]
            j_idx = pair_indices[:, 1]
            valid_pairs = valid_metric[i_idx] & valid_metric[j_idx]
            if valid_pairs.sum() == 0:
                continue
            i_idx = i_idx[valid_pairs]
            j_idx = j_idx[valid_pairs]
            if i_idx.numel() > max_pairs_per_metric:
                perm = torch.randperm(i_idx.numel(), device=device)[:max_pairs_per_metric]
                i_idx = i_idx[perm]
                j_idx = j_idx[perm]

            loss_vec = _pairwise_hinge_rank_loss(
                pred_i=pred_slide[i_idx, metric_index],
                pred_j=pred_slide[j_idx, metric_index],
                true_i=target_slide[i_idx, metric_index],
                true_j=target_slide[j_idx, metric_index],
                margin=margin,
            )
            if loss_vec is None:
                continue

            if use_distance_weight:
                pair_dist2 = dist2[i_idx, j_idx]
                weights = torch.exp(-pair_dist2 / (2.0 * distance_sigma * distance_sigma + eps))
                metric_loss = (loss_vec * weights).sum() / weights.sum().clamp_min(eps)
            else:
                metric_loss = loss_vec.mean()

            total_loss = total_loss + metric_loss
            total_terms += 1

    if total_terms == 0:
        return pred.new_tensor(0.0)
    return total_loss / total_terms
