# STS-2 Oracle Geometry Benchmark

## v0.2.0: adaptive UV transport experiment

v0.2.0 implements the next falsification experiment after the saved v0.1.1 results. It preserves every legacy renderer and adds a separate path that turns transport coherence into an adaptive reflected mesh instead of only reducing root solves.

The new primary comparison is:

```text
analytic reflected-ray reference
        |
        +-- dense exact-root UV microtriangles        exact_tri_uv
        |
        +-- adaptive Taylor UV microtriangles         adaptive_pred_tri_uv
        |
        +-- adaptive Taylor + 1 Newton UV triangles   adaptive_newton_tri_uv
```

The UV paths carry source coordinates and reflected depth, not vertex RGB. First-hit visibility is resolved in `tau`, then the winning source field is sampled per fragment. The same source field and filter are therefore shared by dense exact and adaptive transport.

### What changed mathematically

For each adaptive tile, v0.2.0 solves one exact stationary-path anchor and evaluates `J_us`. Generated vertices use

```text
u_pred = u0 + J_us (s - s0)
```

and the Newton mode applies exactly one undamped production correction

```text
u_1 = u_pred - F_u(u_pred)^-1 F(u_pred).
```

The adaptive criterion deliberately separates five errors:

1. remaining root error, estimated by the *next* Newton displacement projected to screen;
2. screen-warp interpolation error, measured between a corrected probe and the actual piecewise-affine triangle map;
3. reflected-depth interpolation error, measured using the screen-space barycentrics the rasterizer will actually use;
4. inverse-map source-coordinate error, `||s - s_hat||`, at the physical probe pixel;
5. source-radiance error between the physical probe and the raster-reconstructed source coordinate.

This separation matters because one Newton correction can make vertex root error locally `O(h^4)` while an affine screen triangle remains generically `O(h^2)` in its interior.

Tiles also split on physical validity, `kappa(F_u)`, `sigma_min(J_phi)`, small `|det J_phi|`, and local orientation changes. At maximum depth unresolved/singular tiles are reported rather than silently hidden.

### Conforming mesh

Adaptive leaves are triangulated through a shared dyadic source-coordinate lattice. Neighboring leaves share cached boundary vertices. When a coarse leaf borders finer leaves its boundary includes the hanging vertices and is triangulated as a polygon fan, preventing T-junction cracks. Emitted shared vertices use the finest incident tile's predictor before the optional one-step Newton correction. Because those transition fans differ from the cell's simple two-triangle estimator, v0.2.0 also probes the final emitted triangles and reports `adaptive_final_topology_tolerance_violations`. The smoke configuration treats any such violation as a hard failure.

### Appearance

`exact_tri_uv` and both adaptive modes rasterize `(p, tau, s)`. The UV rasterizer returns the winning source coordinate and the affine `ds/dp` footprint. `PlaneSurface.radiance_filtered()` then evaluates the procedural shared appearance field with 1, 4, or 9 deterministic footprint taps. This removes the v0.1.1 coupling between source texture bandwidth and baked vertex RGB.

### New diagnostics

Adaptive runs report, among other fields:

- `exact_anchor_solves`;
- `adaptation_production_newton_corrections`;
- `adaptation_estimator_newton_corrections`;
- `emitted_vertex_newton_corrections`;
- `total_newton_linear_solves`;
- `generated_vertices`, `generated_triangles`, `invalid_triangles`;
- estimated root-screen, triangle-screen, triangle-depth, source-UV, and radiance p50/p95/p99/max errors;
- final emitted-transition-triangle probe errors and tolerance-violation counts;
- optional fully re-solved validation-probe versions of the cell-level errors;
- `max_kappa_fu`, `min_sigma_min_jphi`, `min_abs_det_jphi`;
- `pixel_touches`, `depth_only_pixel_touches`;
- `indexed_mesh_payload_bytes` and `dynamic_raster_payload_bytes`.

`persistent_transport_bytes` is zero for these oracle modes because the adaptive transport mesh is generated per view. The existing payload byte metrics remain working-state diagnostics, not end-to-end NVS model-size claims.

### New run order

Start with the small correctness/configuration run:

```bash
./scripts/run_a1_smoke.sh
```

Then run the actual multi-view/source-distance/curvature/bandwidth sweep:

```bash
./scripts/run_a1.sh
```

The sweep varies camera center, reflector curvature, source distance, and source bandwidth. It writes the usual metrics plus adaptive triangle-count and anchor-count plots.

Only after A1 is credible, run adaptive reflected visibility closure:

```bash
./scripts/run_a2_adaptive.sh
```

The legacy v0.1.1 A0/A2/A3 configurations remain available for regression.

### Timing limitation

The adaptive mesh builder is intentionally a correctness-first Python control implementation. It performs adaptive control flow and mesh assembly on the host and is **not** an optimized throughput implementation. `frame_ms` is useful for detecting pathological work growth, but it must not be used to claim an STS speed win or loss against hardware BVH rays. The hard runtime comparison still requires a fused CUDA/graphics implementation and a genuine RTX/OptiX/DXR-style one-bounce baseline on identical geometry/filtering/visibility semantics.

### Validation status

v0.2.0 was prepared by static inspection and syntax validation only. The v0.1.1 numerical results below are saved prior evidence. No v0.2.0 renderer, test, benchmark, or training job was executed while producing this patch.

---


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
- Legacy RGB-baked warped microtriangles plus the v0.2.0 UV-carrying exact/adaptive microtriangle path.
- Separate reflected-depth buffer, including depth-only occluder surfaces.
- Pure-PyTorch raster fallback.
- Optional CuPy/NVRTC CUDA kernels for EWA and triangle depth/color rasterization.
- Legacy v0.1.1 uniform-tile A1 plus v0.2.0 adaptive UV A1 configurations.
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

### Legacy v0.1.1 A1 configuration

`configs/a1_sweep.yaml` is retained only to reproduce the old uniform-tile experiment. `./scripts/run_a1.sh` now runs `configs/a1_adaptive_sweep.yaml`, described in the v0.2.0 section above. Do not use the legacy A1 CSV as evidence for the adaptive representation.

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
- one-step Newton correction residual reduction;
- UV first-hit rasterization and source-coordinate gradients;
- adaptive indexed-mesh topology sanity;
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
