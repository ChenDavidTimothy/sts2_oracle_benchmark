from __future__ import annotations

from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from PIL import Image


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_image(path: str | Path, image: torch.Tensor) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(p)


def save_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warmup: int = 2, repeats: int = 5):
    for _ in range(warmup):
        fn()
    gpu_sync()
    vals = []
    out = None
    for _ in range(repeats):
        gpu_sync()
        t0 = time.perf_counter()
        out = fn()
        gpu_sync()
        vals.append((time.perf_counter() - t0) * 1000.0)
    vals.sort()
    return out, vals[len(vals) // 2], vals
