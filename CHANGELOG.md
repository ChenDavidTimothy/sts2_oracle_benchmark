# 0.1.1

- Gate EWA and CuPy triangle color by the winning first-hit surface, including near-depth occlusion regression tests.
- Report source-hit and union-hit quality plus direct tiled-versus-exact triangle distortion.
- Extend finite-difference checks to camera and depth/image derivatives; tighten smoke quality checks.
- Install the CUDA 13 CuPy wheel through the CUDA extra.

# 0.1.0

Initial oracle-geometry STS-2 falsification benchmark.

# 0.2.0

- Add adaptive source-chart microtriangles that make transport coherence reduce emitted reflected geometry, not only nonlinear root solves.
- Add exactly one fixed Newton production correction with a separate next-Newton root-error estimator.
- Split adaptive control into root error, screen-warp interpolation error, reflected-depth interpolation error, inverse-map source-UV error, source-radiance error, conditioning, determinant/orientation, and validity gates.
- Build a conforming indexed adaptive mesh with shared dyadic boundary vertices and audit the final hanging-node transition triangles separately from the cell estimator.
- Add `exact_tri_uv`, `adaptive_pred_tri_uv`, and `adaptive_newton_tri_uv` modes.
- Add first-hit UV rasterization for Torch and CuPy. Source RGB is sampled after reflected-depth visibility rather than baked into vertices.
- Add affine `ds/dp` footprints and deterministic 1/4/9-tap shared-source filtering.
- Add optional exact validation probes and p50/p95/p99/max transport/interpolation diagnostics.
- Add adaptive smoke, multi-view/source-distance sweep, visibility configuration, and adaptive plots.
- Preserve all v0.1.1 modes for regression. No v0.2.0 benchmark results are claimed in this patch.
