from __future__ import annotations

from dataclasses import dataclass
import torch

from .camera import PinholeCamera
from .geometry import HeightFieldReflector, PlaneSurface
from .transport import solve_specular, differentials, specular_terms

Tensor = torch.Tensor


@dataclass
class TileTransportResult:
    u: Tensor
    valid: Tensor
    tile_index: Tensor
    anchors_s: Tensor
    anchors_u: Tensor
    j_us: Tensor
    anchor_valid: Tensor
    estimated_pixel_error: Tensor
    exact_anchor_solves: int


def uniform_tile_anchors(tiles_per_axis: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    step = 2.0 / tiles_per_axis
    q = torch.linspace(-1.0 + 0.5 * step, 1.0 - 0.5 * step, tiles_per_axis, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def assign_uniform_tiles(s: Tensor, tiles_per_axis: int) -> Tensor:
    ij = torch.floor((s + 1.0) * 0.5 * tiles_per_axis).to(torch.int64)
    ij = ij.clamp(0, tiles_per_axis - 1)
    return ij[..., 1] * tiles_per_axis + ij[..., 0]


def transport_uniform_tiles(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    s: Tensor,
    tiles_per_axis: int,
) -> TileTransportResult:
    c, _, _, _, _ = camera.matrices(device=s.device, dtype=s.dtype)
    anchors_s = uniform_tile_anchors(tiles_per_axis, device=s.device, dtype=s.dtype)
    sol = solve_specular(reflector, surface, anchors_s, c)
    diff = differentials(reflector, surface, camera, anchors_s, sol.u)
    idx = assign_uniform_tiles(s, tiles_per_axis)
    ds = s - anchors_s[idx]
    u = sol.u[idx] + torch.einsum("...ij,...j->...i", diff.j_us[idx], ds)
    valid = sol.valid[idx] & reflector.in_bounds(u)

    # Residual-based one-Newton-step image-error estimate for each transported point.
    q = specular_terms(reflector, surface, u, s, c)
    reg = 1e-10 * torch.eye(2, device=s.device, dtype=s.dtype).expand_as(q["F_u"])
    du = torch.linalg.solve(q["F_u"] + reg, -q["F"][..., None]).squeeze(-1)
    jpi = camera.jacobian(q["x"])
    dp = torch.einsum("...ij,...jk,...k->...i", jpi, q["jx"], du)
    err = torch.linalg.vector_norm(dp, dim=-1)
    return TileTransportResult(
        u=u,
        valid=valid,
        tile_index=idx,
        anchors_s=anchors_s,
        anchors_u=sol.u,
        j_us=diff.j_us,
        anchor_valid=sol.valid,
        estimated_pixel_error=err,
        exact_anchor_solves=anchors_s.shape[0],
    )
