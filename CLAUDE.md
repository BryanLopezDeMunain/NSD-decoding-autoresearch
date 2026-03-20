# CLAUDE.md

## Project overview

Visual category decoding (24 classes) from fMRI flat maps.
Dataset: [`clane9/nsd-flat-cococlip`](https://huggingface.co/datasets/clane9/nsd-flat-cococlip).
Single training script, single GPU, <=20 min wall time, no additional data.

## Commands

```bash
uv sync
uv run python train_nsd_decoding.py
uv run python train_nsd_decoding.py --subset subj01
uv run python train_nsd_decoding.py --latent_dim 512 --depth 4 --dropout 0.5 --lr 5e-4
```

## Data pipeline

1. Load fMRI flat maps from HuggingFace (uint8, 215x200)
2. Mask background voxels using `metadata/nsd_flat_mask.npy` (18,577 survive)
3. Per-sample z-normalization
4. Learned linear projection to `latent_dim` dimensions
5. ResidualMLP classifier (24 classes)

## Subsets

- `ood` (default): Cross-subject. Train on subjects 0,1,2,5,6,7. Val: subject 3. Test: subject 4.
- `subj01`: Within-subject. Train/val/test all from subject 0 (train, testid, shared1000 splits).

## Architecture

ResidualMLP: Linear projection → residual blocks (LayerNorm → Linear → GELU → Dropout → Linear + skip) → LayerNorm → Linear head.

## Baseline

OOD test accuracy: ~20%. Within-subject (subj01): ~36%. Chance: 4.2%.
