from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out-dir", default=None)
args = ap.parse_args()

df = pd.read_csv(args.csv)
df = df[df.method.isin(["exact_tri_uv", "adaptive_pred_tri_uv", "adaptive_newton_tri_uv"])].copy()
out_dir = Path(args.out_dir or Path(args.csv).parent)
out_dir.mkdir(parents=True, exist_ok=True)

quality = "direct_adaptive_vs_exact_uv_psnr_union_hit"
adaptive = df[df.method.str.startswith("adaptive_")].copy()

if quality in adaptive.columns:
    fig = plt.figure(figsize=(7, 5))
    for method, group in adaptive.groupby("method"):
        plt.scatter(group["generated_triangles"], group[quality], label=method)
    plt.xscale("log")
    plt.xlabel("Generated reflected triangles")
    plt.ylabel("Adaptive vs dense-exact union-hit PSNR (dB)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "adaptive_triangles_vs_exact_quality.png"
    fig.savefig(path, dpi=160)
    print(path)

    fig = plt.figure(figsize=(7, 5))
    for method, group in adaptive.groupby("method"):
        plt.scatter(group["exact_anchor_solves"], group[quality], label=method)
    plt.xscale("log")
    plt.xlabel("Exact anchor solves")
    plt.ylabel("Adaptive vs dense-exact union-hit PSNR (dB)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "adaptive_anchor_solves_vs_exact_quality.png"
    fig.savefig(path, dpi=160)
    print(path)

fig = plt.figure(figsize=(7, 5))
for method, group in df.groupby("method"):
    if "generated_triangles" in group:
        plt.scatter(group["frame_ms"], group["generated_triangles"], label=method)
plt.xlabel("Prototype frame time (ms)")
plt.ylabel("Generated reflected triangles")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
path = out_dir / "adaptive_time_vs_triangles.png"
fig.savefig(path, dpi=160)
print(path)
