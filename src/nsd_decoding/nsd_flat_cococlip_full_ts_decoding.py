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
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parents[2]
SCRIPT = Path(__file__).stem
NUM_CLASSES = 24
NUM_TRS = 16
NUM_VOXELS = 77763

# Build COCO category ID -> contiguous label mapping
COCO_CATEGORIES = json.loads((ROOT / "metadata/nsd_cococlip_categories.json").read_text())
COCO_ID_TO_LABEL = {coco_id: ii for ii, coco_id in enumerate(COCO_CATEGORIES.values())}


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand(x.shape[0], 1, device=x.device) < keep_prob
        return x * mask / keep_prob


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.5, drop_path=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.block(x))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        latent_dim,
        depth,
        num_classes=NUM_CLASSES,
        dropout=0.5,
        drop_path=0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(NUM_VOXELS, latent_dim)
        self.t_proj = nn.Linear(NUM_TRS, latent_dim, bias=False)
        self.blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, NUM_TRS, NUM_VOXELS)
        x = self.proj(x)  # (B, NUM_TRS, latent_dim)
        x = torch.einsum("btd, dt -> bd", x, self.t_proj.weight)  # (B, latent_dim)
        x = self.blocks(x)
        x = self.head(x)
        return x


class BoldDataset(Dataset):
    """Wraps a HF dataset split, filtering by subject and mapping labels."""

    def __init__(self, ds, subs=None):
        if subs:
            sub_strings = [f"subj{s + 1:02d}" for s in subs]
            sub_col = np.array(ds["sub"])
            keep = np.isin(sub_col, sub_strings)
            ds = ds.select(np.where(keep)[0])
        self.ds = ds
        self.ds.set_format("torch")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        bold = sample["bold"].float()  # (16, 77763)
        # Per-frame spatial z-normalize
        mean = bold.mean(dim=-1, keepdim=True)
        std = bold.std(dim=-1, keepdim=True).clamp(min=1e-6)
        bold = (bold - mean) / std
        label = COCO_ID_TO_LABEL[int(sample["category_id"])]
        return bold, label


def train_one_epoch(model, loader, optimizer, criterion, device, steps_per_epoch=25):
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
        x, y = x.to(device), y.to(device)
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
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.append(out.argmax(1).cpu())
        all_targets.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    start_t = time.monotonic()
    sha, is_clean = get_sha()
    print(f"sha: {sha}, clean: {is_clean}")

    if args.ood:
        print("evaluating OOD decoding")
        subs = None
        split_map = {"train": "train", "val": "validation", "test": "test"}
    else:
        print("evaluating ID decoding")
        subs = [int(s) for s in args.subs.split(",")]
        print(f"subjects: {subs}")
        split_map = {"train": "train", "val": "testid", "test": "shared1000"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load data
    dataset_root = ROOT / "datasets/nsd_flat_cococlip_full_ts"

    print("Loading datasets...")
    datasets = {}
    for name, hf_split in split_map.items():
        hf_ds = hfds.load_dataset(
            "arrow",
            data_files=f"{dataset_root}/{hf_split}/*.arrow",
            split="train",
        )
        ds = BoldDataset(hf_ds, subs)
        datasets[name] = ds
        print(f"  {name} ({hf_split}): {len(ds)} samples")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loaders = {
        split: DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
        )
        for split, ds in datasets.items()
    }

    # Model
    model = ResidualMLP(
        latent_dim=args.latent_dim,
        depth=args.depth,
        dropout=args.dropout,
        drop_path=args.drop_path,
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
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        val_preds, val_targets = evaluate(model, eval_loaders["val"], device)
        val_acc = 100 * accuracy_score(val_targets, val_preds)

        elapsed = time.monotonic() - start_t
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | loss={loss:.4f} | train_acc={train_acc:.1f}% | val_acc={val_acc:.1f}% | {elapsed:.0f}s"
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
    for split in datasets:
        preds, targets = evaluate(model, eval_loaders[split], device)
        acc = 100 * accuracy_score(targets, preds)
        scores[f"acc_{split}"] = round(acc, 3)

    result = {
        "script": SCRIPT,
        "args": vars(args),
        "sha": sha,
        "clean": is_clean,
        "wall_t": round(time.monotonic() - start_t, 3),
        "best_epoch": best_epoch,
        **scores,
    }
    print(json.dumps(result))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--subs", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--notes", type=str, default=None)
    args = parser.parse_args()
    main(args)
