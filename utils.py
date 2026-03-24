"""Utils for NSD visual category decoding.

Fixed, do not modify.
"""

import os
import random
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import datasets as hfds
import torch

ROOT = Path(__file__).parent

NSD_NUM_CLASSES = 24

NSD_CATEGORIES = {
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "fire hydrant": 10,
    "stop sign": 11,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "zebra": 22,
    "giraffe": 23,
    "umbrella": 25,
    "skis": 30,
    "snowboard": 31,
    "kite": 33,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "pizza": 53,
    "cake": 55,
    "bed": 59,
    "toilet": 61,
    "clock": 74,
}

TIME_BUDGET = 5 * 60


def load_nsd_flat_mask() -> np.ndarray:
    mask = np.load(ROOT / "nsd_flat_mask.npy")
    return mask


def load_nsd_cococlip(subset: Literal["ood", "subj01"] = "ood") -> dict[str, hfds.Dataset]:
    if subset == "ood":
        split_map = {"train": "train", "val": "validation", "test": "test"}
        subs = None
    elif subset == "subj01":
        split_map = {"train": "train", "val": "testid", "test": "shared1000"}
        subs = [0]
    else:
        raise ValueError(f"invalid {subset=}")

    dataset_dict = hfds.load_dataset("clane9/nsd-flat-cococlip")
    dataset_dict.set_format("numpy")
    dataset_dict_filtered = {}

    for split, hf_split in split_map.items():
        ds = dataset_dict[hf_split]
        if subs is not None:
            subject_ids = np.array(ds["subject_id"])
            keep = np.isin(subject_ids, subs)
            ds = ds.select(np.where(keep)[0])
        dataset_dict_filtered[split] = ds

    return dataset_dict_filtered


def accuracy_score(targets: np.ndarray, preds: np.ndarray) -> float:
    return round(100 * float((preds == targets).mean()), 1)


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


def random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
