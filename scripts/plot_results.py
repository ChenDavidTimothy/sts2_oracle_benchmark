from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default=None)
args=ap.parse_args()
df=pd.read_csv(args.csv)
df=df[df.method != "analytic_ray_reference"].copy()
quality_column = "psnr_ref_radiance_hit" if "psnr_ref_radiance_hit" in df else "psnr_reflect"
fig=plt.figure(figsize=(7,5))
for method,g in df.groupby("method"):
    plt.scatter(g.frame_ms,g[quality_column],label=method)
plt.xlabel("Median frame time (ms)")
plt.ylabel("Reference radiance-hit PSNR (dB)" if quality_column == "psnr_ref_radiance_hit" else "Reflection-region PSNR (dB)")
plt.grid(True,alpha=0.25)
plt.legend()
plt.tight_layout()
out=Path(args.out or (str(Path(args.csv).with_suffix(""))+"_time_psnr.png"))
fig.savefig(out,dpi=160)
print(out)
