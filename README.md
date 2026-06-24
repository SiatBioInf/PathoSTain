# Virtual Staining Model

This repository contains PyTorch code for training and evaluating an attention-based virtual staining model from precomputed image features and spatial spot metadata.

## Repository Contents

- `model.py`: biomarker-token decoder with self-attention and cross-attention.
- `dataset.py`: parquet metadata loading, feature loading, target normalization, and slide-aware sampling.
- `losses.py`: masked MSE loss and local rank loss.
- `train.py`: single-fold training and validation.
- `test.py`: ensemble evaluation from trained fold checkpoints.
- `config.yaml`: example configuration with public, relative paths.

## Data Format

The training metadata should be a parquet file. Required columns are:

- `slide_id`: slide or sample identifier.
- `coord`: coordinate/file stem used to locate the feature tensor.
- `pxl_row_in_fullres`: spot row coordinate in full-resolution pixel space.
- `pxl_col_in_fullres`: spot column coordinate in full-resolution pixel space.

All other numeric columns are treated as prediction targets unless they are listed as non-target metadata columns in `dataset.py`.

Feature tensors are expected at:

```text
{features_dir}/{feature_slide_id}/{coord}.pt
```

If the feature folder name differs from `slide_id`, provide either a `feature_slide_id` column in the parquet file or a `data.slide_id_to_feature_dir` mapping in `config.yaml`.

Each `.pt` feature file may contain a tensor, a NumPy array, or a dictionary with one of the keys `feature`, `feat`, or `x`.

## Installation

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if GPU training is needed.

## Training

Edit `config.yaml` to point to your metadata and feature directories, then run:

```bash
python train.py --config config.yaml
```

The script writes checkpoints, training logs, and target normalization statistics under `train.output_dir`.

For k-fold training, either create one config per fold with different `split.fold_id` values or update the output directory for each run.

## Testing

After training fold checkpoints, run:

```bash
python test.py --config config.yaml --ckpt_dir outputs/train --num_folds 5
```

The test script expects each fold checkpoint at one of these common layouts:

```text
{ckpt_dir}/fold_0/checkpoints/best.pt
{ckpt_dir}/fold0/checkpoints/best.pt
```

Predictions and evaluation metrics are written under `test.output_dir`.

## Public Release Notes

This cleaned version intentionally does not include local machine paths, internal slide IDs, training logs, caches, raw data, or trained weights. If pretrained weights are released, use Git LFS or a model/data repository and document the download location.

Before publishing, confirm the license choice, data-sharing policy, and any repository link required by the journal or institution.
