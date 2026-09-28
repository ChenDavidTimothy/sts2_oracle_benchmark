# STS-2 Oracle Geometry Benchmark

This repository is the **first falsification implementation** of the frozen STS-2 mathematics from the design discussion.

It is deliberately **not** a complete learned 3DGS replacement. It implements the experiment that should be run *before* spending time on source inference, learned reflective geometry, rough BRDFs, or a full CUDA renderer.

The benchmark asks one narrow question:

> With exact surface geometry and known source appearance, can sparse tile-coherent specular transport reproduce first-bounce mirror reflection while using many fewer exact specular solves than per-source transport, and does reflected-depth closure remain accurate when occluders are added?

## What is implemented

- Analytic height-field reflectors: plane, quadratic, and ripple/mixed-curvature.
- Planar rectangular source surfaces with analytic checker/sine/constant radiance.
- Exact batched specular root solve from Fermat's stationary optical path condition.
- Analytic `F_u`, `F_s`, `F_c`, `J_us`, `J_uc`, `J_phi`, reflected depth `tau`, and depth gradient.
- Uniform tiled first-order transport, one exact root per tile anchor.
- Per-pixel analytic reflected-ray reference renderer.
- Experimental EWA source-splat renderer; Gaussian support currently extends beyond bounded source silhouettes.
- Warped microtriangle renderer, the primary oracle path.
- Separate reflected-depth buffer, including depth-only occluder surfaces.
- Pure-PyTorch raster fallback.
- Optional CuPy/NVRTC CUDA kernels for EWA and triangle depth/color rasterization.
- Stage A1 parameter sweeps.
- Stage A3 multistart branch-count diagnostic on mixed-curvature reflectors.
- Whole-mirror and source-hit/union-hit PSNR, direct tiled-versus-exact triangle quality, reflected-hit precision/recall/F1, matched-depth error, first-hit surface/role agreement, timing, root counts, tile error estimate, triangle/sample counts, and raster pixel-touch counts.
- Prototype byte diagnostics: `transport_anchor_payload_bytes` (camera-frame anchor/Jacobian payload proxy) and `dynamic_raster_payload_bytes` (generated raster working-set payload). These are **not** encoded model-byte claims.
- Finite-difference derivative tests, planar virtual-source test, Taylor-order test, and a rendering smoke test.

## What is intentionally not implemented yet

These should **not** be added unless the oracle benchmark survives:

- learned reflector geometry;
- source discovery or latent mirror-only geometry;
- rough BRDF transport;
- recursive mirror bounces;
- learned visibility gates;
- QRFS/free-motion production rendering;
- full Ref-DGS, HybridSplat, SpecTRe-GS, RGS, or 3DGS training code;
- a true RTX/OptiX BVH baseline.

The repository includes an analytic per-reflection-pixel ray reference, not an OptiX performance baseline. If STS-2 survives A0-A3, the next benchmark step is integrating an RTX hardware-ray implementation and current published reflective Gaussian systems. Do not interpret the present ray timing as hardware-RT timing.

## Target machine

Designed for:

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Ti, compute capability 8.6
- current NVIDIA driver compatible with CUDA 13.0 runtime
- Python 3.10-3.12
- PyTorch 2.14.0 + CUDA 13.0 wheel

The PyTorch wheel bundles the CUDA runtime it needs. The CUDA extra installs `cupy-cuda13x[ctk]`, including CUDA components needed by CuPy RawKernel on a machine without a system CUDA toolkit. If CuPy does not initialize, the repository falls back to PyTorch.

## Install

From the extracted directory:

```bash
chmod +x scripts/setup_ubuntu_24_04.sh
./scripts/setup_ubuntu_24_04.sh
```

The setup script:

1. checks `nvidia-smi`;
2. uses the existing `.venv` without changing system packages;
3. installs the official PyTorch 2.14.0 CUDA 13.0 wheel;
4. installs CuPy and benchmark dependencies;
5. checks CUDA/GPU detection;
6. runs the test suite.

After setup:

```bash
source .venv/bin/activate
python -m sts2.check_install
```

For the RTX 3060 Ti, the check should report a CUDA GPU with compute capability `(8, 6)`.

## Run order

### A0: correctness smoke test

```bash
./scripts/run_a0.sh
```

Outputs:

```text
runs/a0_smoke/
  metrics_all.csv
  system.json
  config.yaml
  case_000/
    reference.png
    exact_ewa.png
    tiled_ewa.png
    exact_tri.png
    tiled_tri.png
    *_absdiff.png
    metrics.csv
```

The relevant sanity checks are:

- `exact_tri` should approach the analytic ray reference as source resolution increases;
- `tiled_tri` should approach `exact_tri` as `tiles_per_axis` increases;
- `tiled_*` should use `tiles_per_axis^2` exact anchor solves per source surface instead of one solve per source vertex/sample;
- `depth_agreement` counts missing hits as failures; first-hit surface and role agreement should also be near 1.
- CUDA A0 compares CuPy and Torch hit masks and first-hit IDs, requires EWA touches, and rejects a black reflected source.
- `triangle_msaa: 4` adds a separate 2x2 supersampled/downsampled triangle quality evaluation; its cost is excluded from `frame_ms`.
- `psnr_ref_radiance_hit` measures quality on reference source-hit pixels; `psnr_radiance_union_hit` also includes predicted source coverage. `direct_tiled_vs_exact_tri_*` isolates transport error from source tessellation error. `psnr_reflect` remains a secondary whole-mirror diagnostic.

### A1: curvature, texture bandwidth, tile-density sweep

```bash
./scripts/run_a1.sh
```

This sweeps:

- reflector curvature;
- source texture bandwidth;
- number of transport tiles.

It writes one CSV across the sweep and a time-vs-PSNR plot.

The core question is whether tiled transport retains image quality while reducing exact root solves. Current timings include host synchronization and a prototype thread-per-primitive CuPy rasterizer; they cannot support a solve-versus-ray speed claim. The current A1 sweep does not vary camera position or source distance.

### A2: reflected-depth visibility closure

```bash
./scripts/run_a2.sh
```

This adds a black opaque depth-only surface between the reflector and source. Both the source and the occluder are transported through the same reflection mapping. The reflected-depth buffer should choose the nearest positive `tau`.

If the output only matches the reference when the depth-only occluder is removed, visibility closure failed.

A2 writes depth and first-hit surface ID buffers as NumPy arrays. `depth_only_pixel_touches` counts candidate depth writes from the occluder, including overlap; it is a traffic count rather than a unique-pixel count.

### A3: branch complexity diagnostic

```bash
./scripts/run_a3.sh
```

This uses a rippled mixed-curvature reflector and multistart Newton solves to count distinct stationary specular roots for source anchors across camera positions.

It writes:

```text
runs/a3_branches.csv
```

This is intentionally a slow diagnostic. It is not the production branch-discovery algorithm.

## Important configuration controls

Every benchmark is YAML-driven.

Key fields:

```yaml
scene:
  reflector:
    kind: quadratic      # plane | quadratic | ripple
    patch_half_extent: 1.2
    kx: -0.12
    ky: -0.08
  camera:
    center: [-0.75, 0.12, 2.7]
    width: 192
    height: 192
  surfaces:
    - role: radiance     # contributes reflected color + depth
    - role: depth_only   # contributes reflected depth only
benchmark:
  source_resolution: 64
  tiles_per_axis: 8
  raster_backend: auto   # auto | torch | cupy
  repeats: 5
```

`raster_backend: auto` selects CuPy on CUDA when available, otherwise PyTorch.

## Mathematical core implemented

For reflector `X(u)` and source `Y(s)`:

```text
F(u;s,c) = J_X(u)^T (ell + d) = 0
ell = (Y-X)/||Y-X||
d   = (c-X)/||c-X||
```

For the height-field chart `X=[u,v,f(u,v)]`:

```text
F_u = H_X(ell+d) - J_X^T [P_ell/r_y + P_d/r_c] J_X
F_s = J_X^T P_ell/r_y J_Y
F_c = J_X^T P_d/r_c
J_us = -F_u^{-1} F_s
J_uc = -F_u^{-1} F_c
```

The exact local reflected image warp derivative is:

```text
J_phi = J_pi J_X J_us
```

Reflected first-hit depth is:

```text
tau = ||Y-X||
J_tau_s = ell^T (J_Y - J_X J_us)
```

The tiled approximation is:

```text
u(s) ~= u0 + J_us(s0) (s-s0)
```

`tests/test_derivatives.py` checks the analytic derivatives against centered finite differences. `tests/test_taylor.py` checks the expected second-order local error of the first-order transport map.

## Reading the results correctly

A successful A0/A1 result means only:

> Under exact geometry and known source appearance, the physical transport map is numerically correct and transport coherence reduces exact path solves without unacceptable distortion.

It does **not** establish:

- a memory win against an end-to-end 3DGS system;
- a runtime win against RTX hardware ray tracing;
- a learned NVS system;
- publication novelty.

A2 is required because the forward transport idea is weak if reflected-depth closure becomes dense or inaccurate. A3 is required because branch multiplicity can destroy the sparse-graph assumption.

Only if A0-A3 pass should you build the next repository stage with learned source inference, learned reflector geometry, full byte accounting, current reflective Gaussian baselines, and hardware RT.

## CPU fallback

For debugging only:

```bash
source .venv/bin/activate
python -m sts2.bench --config configs/a0_smoke.yaml --device cpu --output runs/cpu_debug
```

The triangle fallback intentionally favors simplicity over speed and can be slow at larger resolutions.

## Tests

```bash
pytest -q
```

Current tests cover:

- analytic `F_u` and `F_s` vs finite differences;
- implicit `J_us` vs re-solved roots;
- exact planar reflection point vs mirrored-source construction;
- `O(h^2)` first-order transport error;
- small exact triangle render vs analytic ray reference.

## If the CuPy backend fails

First verify:

```bash
nvidia-smi
source .venv/bin/activate
python -m sts2.check_install
python - <<'PY'
import cupy as cp
print(cp.show_config())
print(cp.cuda.runtime.getDeviceCount())
PY
```

You can force the fallback by changing:

```yaml
raster_backend: torch
```

The benchmark will still be useful for correctness/rate-distortion, but not for a final throughput conclusion.
