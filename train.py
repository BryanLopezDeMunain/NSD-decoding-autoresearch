"""
Autoresearch training script for NSD visual category decoding.
Adapted from nanochat framework. Single-GPU, single-file.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets as hfds
from torch.utils.data import DataLoader, TensorDataset

from prepare import TIME_BUDGET

NSD_ROOT = Path("/home/bryanlopez/nsd-decoding")
NUM_CLASSES = 24

# ---------------------------------------------------------------------------
# NSD Decoding Model
# ---------------------------------------------------------------------------

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
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.depth = depth
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

    def setup_optimizer(self, lr=1e-3, weight_decay=0.01, adam_betas=(0.9, 0.999)):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                      weight_decay=weight_decay, betas=adam_betas)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def evaluate_val(model, loader, device):
    """Returns (cross_entropy_loss, accuracy_pct)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += F.cross_entropy(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 6               # number of residual blocks
LATENT_DIM = 512        # hidden dimension of ResidualMLP
DROPOUT = 0.1           # dropout rate

# Data
SUBSET = "ood"          # "ood" (cross-subject) or "subj01" (within-subject)
BATCH_SIZE = 256        # training batch size

# Optimization
LR = 1e-3               # base learning rate
WEIGHT_DECAY = 0.1      # weight decay
ADAM_BETAS = (0.999, 0.999) # AdamW betas
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.0    # no cooldown; val checkpointing handles early stopping
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# ---------------------------------------------------------------------------
# Setup: data, model, optimizer
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mask
mask = np.load(NSD_ROOT / "metadata/nsd_flat_mask.npy")
num_voxels = int(mask.sum())
print(f"Mask: {num_voxels} voxels out of {mask.size}")

# Load NSD dataset
if SUBSET == "ood":
    print("Loading OOD (cross-subject) data")
    split_map = {"train": "train", "val": "validation", "test": "test"}
    subs = None
else:
    print("Loading subj01 (within-subject) data")
    split_map = {"train": "train", "val": "testid", "test": "shared1000"}
    subs = [0]

print("Loading NSD dataset from HuggingFace...")
dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")

splits = {}
for name, hf_split in split_map.items():
    act, tgt = load_split_tensors(dataset_dict[hf_split], mask, subs)
    splits[name] = (act.to(device), tgt.to(device))
    print(f"  {name} ({hf_split}): {act.shape}, targets: {tgt.shape}")

train_loader = DataLoader(
    TensorDataset(*splits["train"]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)
eval_loaders = {
    split: DataLoader(TensorDataset(act, tgt), batch_size=512)
    for split, (act, tgt) in splits.items()
}

# Build model
model = ResidualMLP(
    input_dim=num_voxels,
    latent_dim=LATENT_DIM,
    depth=DEPTH,
    dropout=DROPOUT,
).to(device)
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params:,}")
print(f"LATENT_DIM: {LATENT_DIM}, DEPTH: {DEPTH}")

optimizer = model.setup_optimizer(lr=LR, weight_decay=WEIGHT_DECAY, adam_betas=ADAM_BETAS)

print(f"Time budget: {TIME_BUDGET}s")

# LR schedule (warmup → flat → cosine cooldown)
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Training loop (time-budget based)
# ---------------------------------------------------------------------------

EVAL_EVERY = 25         # checkpoint every N steps (eval time not counted in budget)

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
train_iter = iter(train_loader)
best_val_bpb = float('inf')
best_state = None

while True:
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    # Random input masking (keep 85% of voxels)
    x_aug = x * (torch.rand_like(x) > 0.15).float()
    logits = model(x_aug)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)

    train_loss = loss.detach()
    loss.backward()

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    # Periodic validation checkpoint (eval time not counted in time budget)
    if step > 0 and step % EVAL_EVERY == 0:
        v_loss, _ = evaluate_val(model, eval_loaders["val"], device)
        v_bpb = v_loss / math.log(2)
        if v_bpb < best_val_bpb:
            best_val_bpb = v_bpb
            best_state = {k: v.cpu().clone() for k, v in model._orig_mod.state_dict().items()}

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_samples = step * BATCH_SIZE

# Load best checkpoint if we found one better than the final state
if best_state is not None:
    model._orig_mod.load_state_dict(best_state)
    model._orig_mod.to(device)

# Final eval
val_loss, val_acc = evaluate_val(model, eval_loaders["val"], device)
val_bpb = val_loss / math.log(2)  # cross-entropy in bits per prediction
_, test_acc = evaluate_val(model, eval_loaders["test"], device)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"val_acc:          {val_acc:.3f}")
print(f"test_acc:         {test_acc:.3f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      0.00")
print(f"total_tokens_M:   {total_samples / 1e6:.3f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
