# v1.1 correctness validation

Validated locally on an RTX 3060 Ti with PyTorch 2.14.0+cu130 and `cupy-cuda13x` 14.2.0. CuPy was switched from the CUDA 12 wheel before the final runs.

- `python -m pytest -q`: 16 passed. Near-depth depth-only occlusion is tested for Torch CPU, Torch CUDA, and CuPy CUDA; CUDA backend parity is checked on the same overlapping primitive buffers.
- A0: CuPy/Torch hit masks and first-hit IDs agree exactly for all methods, with color differences below `4e-7`. Exact triangle quality is 53.47 dB over the mirror and 41.75 dB on reference source-hit pixels. Exact EWA is 22.42 dB over the mirror and has 0.822 radiance-hit precision, reflecting excess Gaussian footprint coverage.
- A0 direct tiled-versus-exact triangles: 64.67 dB on common radiance hits, 29.57 dB on their union, with two radiance-mask disagreement pixels. The 4x supersampled union-hit score is 46.61 dB; the corresponding whole-mirror score is 58.17 dB. These masks expose the boundary effect hidden by whole-mirror PSNR.
- A2: CuPy/Torch backend parity passes. Exact triangles have hit precision, hit recall, and first-hit surface agreement of 1.0 in this one scene. Exact EWA still has 0.895 hit precision and 0.937 first-hit surface agreement; first-surface color gating does not correct its footprint/visibility approximation.
- A2 depth-only occluder traffic: 1,127 triangle touches and 28,231 EWA touches before mirror-mask clipping. Touches count candidate writes, not unique pixels.

Reproduce the checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m sts2.bench --config configs/a0_smoke.yaml --device cuda --output runs/a0_v1_1_cuda
.venv/bin/python -m sts2.bench --config configs/a2_visibility.yaml --device cuda --output runs/a2_v1_1_cuda
```

The per-case directories contain image, depth, first-hit surface ID, and CSV artifacts. CUDA A0/A2 fail on backend mask, ID, or color disagreement. The run's `frame_ms` includes host/device barriers and a prototype thread-per-primitive rasterizer, so it is **not** solve-versus-ray or optimized-renderer performance evidence. The A1 multi-view/source-distance sweep and adaptive transport remain future work.

# v0.2.0 validation status

The v0.2.0 adaptive UV patch has **not** been numerically executed as part of its preparation. The constraint for this revision was to inspect and modify the code without rerunning the renderer/benchmark. Static Python compilation and patch-integrity checks may be performed, but any numerical or CUDA result must come from the target RTX 3060 Ti run and must be recorded separately.

The first required runtime validation is `configs/a1_adaptive_smoke.yaml`. Do not interpret `frame_ms` as an optimized renderer comparison: the adaptive builder is correctness-first Python control code. The first hard claims to inspect are dense-exact versus adaptive UV image agreement, direct first-hit agreement, exact-anchor/triangle/payload reduction factors, estimator calibration, zero final-topology tolerance violations, and CuPy/Torch UV-raster parity.
