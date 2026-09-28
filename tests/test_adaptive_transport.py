import torch

from sts2.adaptive import AdaptiveTransportConfig, _Cell, _evaluate_cell, build_adaptive_transport_mesh
from sts2.camera import PinholeCamera
from sts2.geometry import HeightFieldReflector, PlaneSurface
from sts2.transport import newton_correct, solve_specular, differentials, specular_terms


def test_one_fixed_newton_correction_reduces_taylor_residual():
    dtype = torch.float64
    refl = HeightFieldReflector(kind="quadratic", kx=-0.2, ky=-0.1, patch_half_extent=1.5)
    surf = PlaneSurface("src", (0.8, 0.0, 2.1), (0.7, 0.0, 0.0), (0.0, 0.6, 0.0))
    cam = PinholeCamera(center=(-0.75, 0.15, 2.7), width=64, height=64)
    c = torch.tensor(cam.center, dtype=dtype)
    s0 = torch.tensor([[0.0, 0.0]], dtype=dtype)
    anchor = solve_specular(refl, surf, s0, c)
    j = differentials(refl, surf, cam, s0, anchor.u).j_us[0]
    s = torch.tensor([[0.18, -0.13]], dtype=dtype)
    pred = anchor.u + torch.einsum("ij,nj->ni", j, s - s0)
    before = torch.linalg.vector_norm(specular_terms(refl, surf, pred, s, c)["F"], dim=-1)
    corrected, _ = newton_correct(refl, surf, s, c, pred)
    after = torch.linalg.vector_norm(specular_terms(refl, surf, corrected, s, c)["F"], dim=-1)
    assert after.item() < before.item() * 0.1


def test_adaptive_mesh_is_indexed_and_conforming_in_unsplit_case():
    refl = HeightFieldReflector(kind="plane", patch_half_extent=2.0)
    surf = PlaneSurface("src", (0.8, 0.0, 2.0), (0.6, 0.0, 0.0), (0.0, 0.5, 0.0))
    cam = PinholeCamera(center=(-0.8, 0.1, 2.7), width=64, height=64)
    cfg = AdaptiveTransportConfig(
        base_tiles_per_axis=2,
        max_depth=0,
        screen_error_px=1e6,
        depth_error=1e6,
        root_error_px=1e6,
        source_uv_error=1e6,
        radiance_error=1e6,
        max_kappa_fu=1e12,
        min_sigma_jphi=0.0,
        min_abs_det_jphi=0.0,
        validate_exact_probes=False,
    )
    mesh = build_adaptive_transport_mesh(refl, surf, cam, cfg, device=torch.device("cpu"), dtype=torch.float64)
    assert mesh.leaf_count == 4
    assert mesh.source_vertices.shape[0] == 9
    assert mesh.triangles.shape[0] == 8
    assert bool(mesh.triangle_valid.all())
    assert mesh.stats["exact_anchor_solves"] == 4
    assert mesh.stats["adaptive_tolerance_exhausted_tiles"] == 0
    assert mesh.stats["adaptive_final_topology_tolerance_violations"] == 0


def test_curved_edge_probe_can_be_outside_affine_triangle_within_error_gates():
    refl = HeightFieldReflector(kind="quadratic", patch_half_extent=1.15, kx=-0.16, ky=-0.10)
    surf = PlaneSurface("src", (0.78, 0.02, 2.05), (0.72, 0.0, 0.0), (0.0, 0.62, 0.0), texture="sine", bandwidth=8.0)
    cam = PinholeCamera(center=(-0.72, 0.15, 2.65), width=128, height=128)
    cfg = AdaptiveTransportConfig(base_tiles_per_axis=4, max_depth=3)
    center = torch.tensor(cam.center, dtype=torch.float64)
    leaf = _evaluate_cell(refl, surf, cam, center, _Cell(62, 62, 2, 3), 64, cfg)
    assert not leaf.screen_probe_inside
    assert leaf.max_screen_interp_error_px < cfg.screen_error_px
    assert leaf.max_source_uv_error < cfg.source_uv_error
    assert leaf.tolerance_met
