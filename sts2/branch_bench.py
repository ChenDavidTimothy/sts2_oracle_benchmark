from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml

from .geometry import make_source_grid
from .scenes import scene_from_config
from .transport import multistart_roots
from .utils import seed_all


def main():
    ap = argparse.ArgumentParser(description="Stage A3 multistart branch diagnostic")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/a3_branches.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_all(int(cfg.get("seed", 1234)))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    scene = scene_from_config(cfg["scene"])
    source = next(s for s in scene.surfaces if s.role == "radiance")
    bcfg = cfg.get("branch_benchmark", {})
    n = int(bcfg.get("anchor_grid", 5))
    s = make_source_grid(n, device=device, dtype=dtype, vertices=False)
    seeds = int(bcfg.get("seeds_per_axis", 7))
    cam_x_values = bcfg.get("camera_x", [scene.camera.center[0]])
    rows = []
    for cx in cam_x_values:
        center = (float(cx), scene.camera.center[1], scene.camera.center[2])
        c = torch.tensor(center, device=device, dtype=dtype)
        roots = multistart_roots(scene.reflector, source, s, c, seeds_per_axis=seeds)
        for i, rr in enumerate(roots):
            rows.append({
                "camera_x": float(cx),
                "source_anchor": i,
                "s_x": float(s[i,0].cpu()),
                "s_y": float(s[i,1].cpu()),
                "root_count": len(rr),
                "roots": ";".join(f"{float(r[0].cpu()):.6f},{float(r[1].cpu()):.6f}" for r in rr),
            })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby("camera_x")["root_count"].agg(["mean","max","sum"]).to_string())
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
