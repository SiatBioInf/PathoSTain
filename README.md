# PathoTME

**PathoTME** is a PyTorch framework for **virtual TME profiling from pathology**. It predicts spot-level tumor microenvironment (TME) molecular and spatial profiles from pathology-derived features.

This project is not an image-to-image translation model. Instead of generating output images, PathoTME performs **spot-level multi-target regression**:

```text
H&E pathology image
  -> pathology foundation model spot-level feature
  -> spatial transcriptomics / biomarker / TME score prediction
```

The current implementation uses precomputed pathology foundation model features, spatial spot metadata, and a biomarker-token decoder to predict spatial transcriptomics-derived biomarkers, protein markers, cell-state scores, pathway scores, and tissue-architecture-related scores.

## Model Overview

PathoTME is designed for pathology-based TME profiling under spatial omics supervision.

Main components:

- **Input**: precomputed spot-level pathology features, such as 1536-dimensional H-optimus features.
- **Spatial metadata**: spot coordinates and derived spatial features.
- **Decoder**: biomarker-token decoder with self-attention and cross-attention.
- **Output**: one scalar prediction for each biomarker, gene score, cell-state score, pathway score, or TME-related target.
- **Losses**: masked MSE loss and local rank loss.
- **Evaluation**: MAE, RMSE, Pearson correlation, Spearman correlation, and C-index.

## Repository Contents

- `model.py`: PathoTME biomarker-token decoder.
- `dataset.py`: parquet metadata loading, feature loading, target normalization, spatial feature construction, and slide-aware sampling.
- `losses.py`: masked MSE loss and local rank loss.
- `train.py`: single-fold training and validation.
- `test.py`: ensemble evaluation from trained fold checkpoints.
- `config.yaml`: example configuration with public, relative paths.
- `requirements.txt`: pinned Python package versions for reproducibility.

## Data Format

The training metadata should be stored as a parquet file. Required columns are:

- `slide_id`: slide or sample identifier.
- `coord`: coordinate or file stem used to locate the feature tensor.
- `pxl_row_in_fullres`: spot row coordinate in full-resolution pixel space.
- `pxl_col_in_fullres`: spot column coordinate in full-resolution pixel space.

All other numeric columns are treated as prediction targets unless they are listed as non-target metadata columns in `dataset.py`. These targets can include spatial transcriptomics-derived biomarkers, protein markers, cell-state scores, pathway scores, or other TME-related continuous annotations.

Precomputed pathology feature tensors are expected at:

```text
{features_dir}/{feature_slide_id}/{coord}.pt
```

If the feature folder name differs from `slide_id`, provide either a `feature_slide_id` column in the parquet file or a `data.slide_id_to_feature_dir` mapping in `config.yaml`.

Each `.pt` feature file may contain a tensor, a NumPy array, or a dictionary with one of the following keys:

```text
feature
feat
x
```

## Installation

The code was organized for Python 3.10.

```bash
pip install -r requirements.txt
```

If GPU training is required, install the PyTorch build that matches your CUDA version. The pinned `torch` version in `requirements.txt` can be replaced by the corresponding CUDA wheel from the official PyTorch installation guide.

## Training

Edit `config.yaml` to point to your metadata and feature directories:

```bash
python train.py --config config.yaml
```

The training script writes checkpoints, logs, and target normalization statistics under `train.output_dir`.

For k-fold training, create one config per fold with different `split.fold_id` values or update the output directory for each run.

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

## Recommended Metrics

PathoTME is a spot-level multi-target regression model. Recommended primary metrics are:

- MAE
- RMSE
- Pearson correlation
- Spearman correlation
- C-index

Optional spatial consistency metrics, such as Moran's I or nearest-neighbor correlation, can be added for evaluating spatial structure preservation.

## Public Release Notes

This cleaned release does not include local machine paths, internal slide IDs, training logs, caches, raw data, or trained weights.

If pretrained weights are released, use Git LFS or a model/data repository and document the download location.

Before publishing, confirm the license choice, data-sharing policy, and any repository link required by the journal or institution.
