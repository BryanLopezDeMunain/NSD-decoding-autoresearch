# NSD Decoding

Visual category decoding (24 classes) from fMRI data using the [Natural Scenes Dataset](https://naturalscenesdataset.org/).

## Task

Given fMRI cortical flat map activity, predict which of 24 visual categories a subject is viewing.

**Dataset:** [`clane9/nsd-flat-cococlip`](https://huggingface.co/datasets/clane9/nsd-flat-cococlip)

Each sample contains:
- `activity`: fMRI flat map (215 x 200, uint8)
- `target`: category label (0-23)
- `subject_id`, `trial_id`, `nsd_id`

## Subsets

### OOD (cross-subject, default)

Train on 6 subjects, evaluate on 2 held-out subjects.

| Split | Notes | Samples |
|-------|----------|---------|
| train | subj 1-3, 6-8 | 32,539 |
| val | subj 4 | 5,418 |
| test | subj 5 | 5,390 |

### subj01 (within-subject)

Train and evaluate on subj01 only.

| Split | Notes | Samples |
|-------|--------|---------|
| train | ses 1-35 | 6,190 |
| val | ses 36-40 | 864 |
| test | shared1000 | 559 |

## Categories

motorcycle, airplane, bus, train, fire hydrant, stop sign, horse, sheep,
cow, elephant, zebra, giraffe, umbrella, skis, snowboard, kite,
skateboard, surfboard, tennis racket, pizza, cake, bed, toilet, clock

## Setup

```bash
uv sync
```

## Run

```bash
# Cross-subject decoding (default)
uv run python train.py

# Within-subject decoding
uv run python train.py --subset subj01

# Custom hyperparameters
uv run python train.py --latent_dim 512 --depth 4 --dropout 0.5 --lr 5e-4
```

On first run, the dataset is downloaded from HuggingFace (~800MB) and cached locally.

## Constraints

- Single GPU
- Wall time at most 5 minutes per run
- No additional data beyond the provided dataset

## Results

### OOD

| Commit | Notes | Test Acc (%) | Wall Time (s) |
| --- | --- | --- | --- |
| [`clane9/b919e73`](https://github.com/clane9/nsd-decoding/tree/b919e73) | Baseline MLP | 26.7 | 0.6 |

### subj01

| Commit | Notes | Test Acc (%) | Wall Time (s) |
| --- | --- | --- | --- |
| [`clane9/b919e73`](https://github.com/clane9/nsd-decoding/tree/b919e73) | Baseline MLP | 62.3 | 2.6 |
