# autoresearch — NSD Decoding

An autonomous research agent for fMRI visual category decoding, forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The agent iterates on `train.py` overnight — modifying architecture and hyperparameters, training for 5 minutes, checking if validation loss improved, and keeping or discarding changes.

## Task

Decode visual categories from fMRI brain activity using the [NSD](https://huggingface.co/datasets/clane9/nsd-flat-cococlip) dataset. The model maps voxel activations to one of 24 COCO super-categories via a residual MLP.

- **Input:** flattened fMRI voxels (masked to visual cortex), z-normalized per sample
- **Model:** `ResidualMLP` — linear projection → residual blocks (LayerNorm + Linear + GELU) → classification head
- **Metric:** `val_bpb` (cross-entropy / log 2) — lower is better

## Files

```
prepare.py      — fixed constants, evaluation harness (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Quick start

**Requirements:** Single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/), NSD data at `~/nsd-decoding/`.

```bash
uv sync
uv run train.py
```

## Running the agent

Open Claude Code in this repo, then:

```
Have a look at program.md and let's kick off a new experiment.
```

## License

MIT
