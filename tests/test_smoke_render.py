import torch

from sts2.camera import PinholeCamera
from sts2.geometry import HeightFieldReflector, PlaneSurface
from sts2.ray_reference import render_reference
from sts2.render import render_triangles
from sts2.metrics import psnr


def test_small_exact_triangle_render_matches_reference_reasonably():
    device = torch.device("cpu")
    refl = HeightFieldReflector(kind="quadratic", kx=-0.10, ky=-0.07, patch_half_extent=1.1)
    cam = PinholeCamera(center=(-0.7, 0.12, 2.6), width=48, height=48, fov_y_deg=44)
    surf = PlaneSurface("src", (0.75, 0.0, 2.0), (0.7, 0, 0), (0, 0.6, 0), texture="sine", bandwidth=5.0)
    ref = render_reference(refl, cam, [surf], device=device, dtype=torch.float64)
    out = render_triangles(refl, cam, [surf], ref.mirror_mask, n_cells=18, mode="exact", backend="torch")
    score = psnr(out.image.double(), ref.image, ref.mirror_mask)
    source_hit_score = psnr(out.image.double(), ref.image, torch.isfinite(ref.first_hit_tau))
    assert score > 45.0
    assert source_hit_score > 35.0
