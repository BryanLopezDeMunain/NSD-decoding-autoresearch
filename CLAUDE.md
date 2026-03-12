# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Visual category decoding (24 classes) from fMRI data using the Natural Scenes Dataset. Models predict a target category from cortical flat map activity. Dataset: [`clane9/nsd-flat-cococlip`](https://huggingface.co/datasets/clane9/nsd-flat-cococlip).

Constraints: single H100 GPU, wall time at most 20 minutes per run, no additional data. Track results in `RESULTS.md`.

## Commands

```bash
# Install (uses uv for package management)
uv sync

# Run a training script
uv run python src/nsd_decoding/nsd_flat_cococlip_decoding_v4.py --n_components 96 --depth 6 --dropout 0.7 --drop_path 0.2 --lr 5e-4

# Launch a sweep via Slurm
sbatch experiments/sweep_v4/launch.sh

# Lint
uv run ruff check src/
uv run ruff format --check src/
```

## Data splits

The validation and test splits contain held-out subjects. The task is cross-subject decoding.

- Train: subjects 0, 1, 2, 5, 6, 7 (32,539 samples)
- Validation: subject 3 (5,418 samples)
- Test: subject 4 (5,390 samples)
- Testid: subjects 0, 1, 2, 5, 6, 7 held-out trials (5,187 samples)

## Preprocessing pipeline

1. Mask background voxels (value 127) using `metadata/nsd_flat_mask.npy` (18,577 of 43,000 survive)
2. Per-sample z-normalization of masked voxels
3. PCA projection + whitening using `datasets/nsd_flat_pca.npz` (fit on training data only)
   - `components`: (512, 18577), `mean`: (18577,), `scale`: (512,) for whitening

## Architecture

Each script version (`src/nsd_decoding/nsd_flat_cococlip_decoding_v{0,1,2,3,4}.py`) is a self-contained, single-file training script with model definition, data loading, training loop, and evaluation. No shared library code — each version is frozen for reproducibility.

- **v4** (current best): ResidualMLP on PCA-projected features. Residual blocks with LayerNorm, GELU, Dropout, DropPath. Trained with AdamW + cosine annealing.
- **id_v1/v2**: Within-subject decoding variants using the `shared1k` split.

Scripts output a JSON result line at the end with accuracies and config, which gets collected into `experiments/*/result.jsonl`.

## Key files

- Training scripts: `src/nsd_decoding/nsd_flat_cococlip_decoding_v*.py`
- Preprocessing notebooks: `notebooks/nsd_flat_masking.ipynb`, `notebooks/nsd_flat_pca.ipynb`
- Metadata: `metadata/nsd_flat_mask.npy`, `datasets/nsd_flat_pca.npz`, `metadata/nsd_cococlip_categories.json`
- Sweep configs: `experiments/sweep_v4/`, `experiments/sweep_v4_nc/`, `experiments/sweep_id_v2/`

## Current status

Best cross-subject test accuracy: 27.8% (v4, n_components=96, depth=6, dropout=0.7, drop_path=0.2, lr=5e-4). PCA dimensionality is the dominant factor for cross-subject generalization. In-distribution and out-of-distribution performance are anticorrelated across n_components.

## Collaboration notes

- Connor prefers to run training scripts himself and watch the logs
- Be direct; minimal style (avoid excessive emojis and emphatic language)
- Each script version is a separate file (v0.py, v1.py, ...) for reproducibility
- Use sbatch array jobs for sweeps (see `experiments/` for the pattern)
- Ruff config: line-length 100, ignore F722 (for jaxtyping annotations)
