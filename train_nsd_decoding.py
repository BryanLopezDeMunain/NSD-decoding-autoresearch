"""Baseline training script for NSD visual category decoding."""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import datasets as hfds
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent
NUM_CLASSES = 24


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, latent_dim, depth, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def load_split_tensors(ds, mask, subs=None):
    """Load a HF dataset split, filter by subjects, apply mask, per-sample z-normalize."""
    if subs is not None:
        subject_ids = np.array(ds["subject_id"])
        keep = np.isin(subject_ids, subs)
        ds = ds.select(np.where(keep)[0])

    activity = np.array(ds["activity"])
    if activity.ndim == 4:
        activity = activity[:, 0]
    activity = activity[:, mask]
    activity = torch.tensor(activity, dtype=torch.float32)

    mean = activity.mean(dim=1, keepdim=True)
    std = activity.std(dim=1, keepdim=True).clamp(min=1e-6)
    activity = (activity - mean) / std

    targets = torch.tensor(np.array(ds["target"]), dtype=torch.long)
    return activity, targets


def train_one_epoch(model, loader, optimizer, criterion, steps_per_epoch=25):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loader_iter = iter(loader)
    for _ in range(steps_per_epoch):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, 100 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    for x, y in loader:
        out = model(x)
        all_preds.append(out.argmax(1))
    return torch.cat(all_preds).cpu().numpy()


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    clean = True
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        clean = not _run(["git", "diff-index", "HEAD"])
    except Exception:
        pass
    return sha, clean


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_t = time.monotonic()
    sha, is_clean = get_sha()
    print(f"sha: {sha}, clean: {is_clean}")

    if args.subset == "ood":
        print("Evaluating OOD (cross-subject) decoding")
        split_map = {"train": "train", "val": "validation", "test": "test"}
        subs = None
    else:
        print("Evaluating subj01 (within-subject) decoding")
        split_map = {"train": "train", "val": "testid", "test": "shared1000"}
        subs = [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load mask
    mask = np.load(ROOT / "metadata/nsd_flat_mask.npy")
    num_voxels = int(mask.sum())
    print(f"Mask: {num_voxels} voxels out of {mask.size}")

    # Load data
    dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")

    print("Loading tensors...")
    splits = {}
    for name, hf_split in split_map.items():
        act, tgt = load_split_tensors(dataset_dict[hf_split], mask, subs)
        splits[name] = (act.to(device), tgt.to(device))
        print(f"  {name} ({hf_split}): {act.shape}, targets: {tgt.shape}")

    train_loader = DataLoader(
        TensorDataset(*splits["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loaders = {
        split: DataLoader(TensorDataset(act, tgt), batch_size=512)
        for split, (act, tgt) in splits.items()
    }

    # Model
    model = ResidualMLP(
        input_dim=num_voxels,
        latent_dim=args.latent_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch = 0
    best_state = None
    for epoch in range(args.epochs):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()

        val_preds = evaluate(model, eval_loaders["val"])
        val_targets = splits["val"][1].cpu().numpy()
        val_acc = 100 * (val_preds == val_targets).mean()

        elapsed = time.monotonic() - start_t
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | loss={loss:.4f} "
            f"| train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_acc < 0.99 * best_val_acc:
            print(f"Early stopping: val_acc {val_acc:.1f}% < 99% of best {best_val_acc:.1f}%")
            break

    # Load best model and evaluate all splits
    model.load_state_dict(best_state)
    model.to(device)

    scores = {}
    for split, (_, targets) in splits.items():
        preds = evaluate(model, eval_loaders[split])
        acc = 100 * (preds == targets.cpu().numpy()).mean()
        scores[f"acc_{split}"] = round(float(acc), 3)

    result = {
        "subset": args.subset,
        "args": vars(args),
        "sha": sha,
        "clean": is_clean,
        "wall_t": round(time.monotonic() - start_t, 3),
        "best_epoch": best_epoch,
        **scores,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="ood", choices=["ood", "subj01"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
