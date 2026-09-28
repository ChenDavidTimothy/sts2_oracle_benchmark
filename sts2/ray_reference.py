from __future__ import annotations

from dataclasses import dataclass
import torch

from .camera import PinholeCamera
from .geometry import HeightFieldReflector, PlaneSurface

Tensor = torch.Tensor


@dataclass
class RayReferenceResult:
    image: Tensor
    mirror_mask: Tensor
    mirror_u: Tensor
    first_hit_tau: Tensor
    first_hit_surface: Tensor


def intersect_reflector(
    reflector: HeightFieldReflector,
    camera: PinholeCamera,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_iter: int = 20,
) -> tuple[Tensor, Tensor, Tensor]:
    o, d = camera.rays(device=device, dtype=dtype)
    # Seed by tangent plane z=0.
    denom = d[..., 2]
    safe = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    t = (-o[..., 2] / safe).clamp_min(1e-4)
    for _ in range(max_iter):
        p = o + t[..., None] * d
        uv = p[..., :2]
        f, grad, _ = reflector.f_grad_hess(uv)
        g = p[..., 2] - f
        dg = d[..., 2] - grad[..., 0] * d[..., 0] - grad[..., 1] * d[..., 1]
        dg = torch.where(dg.abs() < 1e-9, torch.full_like(dg, 1e-9), dg)
        dt = g / dg
        t = (t - dt).clamp_min(1e-5)
        if float(dt.abs().max()) < 1e-7:
            break
    p = o + t[..., None] * d
    uv = p[..., :2]
    f, _, _ = reflector.f_grad_hess(uv)
    residual = (p[..., 2] - f).abs()
    mask = (t > 0) & reflector.in_bounds(uv) & (residual < 5e-5)
    return p, uv, mask


def intersect_plane_surface(x: Tensor, r: Tensor, surface: PlaneSurface) -> tuple[Tensor, Tensor, Tensor]:
    c, au, av = surface.tensors(device=x.device, dtype=x.dtype)
    n = torch.nn.functional.normalize(torch.cross(au, av, dim=0), dim=0)
    denom = (r * n).sum(dim=-1)
    safe = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    t = ((c - x) * n).sum(dim=-1) / safe
    p = x + t[..., None] * r
    s = surface.coordinates(p)
    valid = (t > 1e-5) & surface.in_bounds(s) & (denom.abs() > 1e-8)
    return t, s, valid


def render_reference(
    reflector: HeightFieldReflector,
    camera: PinholeCamera,
    surfaces: list[PlaneSurface],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    environment_rgb: tuple[float, float, float] = (0.03, 0.04, 0.06),
) -> RayReferenceResult:
    x, u, mirror_mask = intersect_reflector(reflector, camera, device=device, dtype=dtype)
    c, _, _, _, _ = camera.matrices(device=device, dtype=dtype)
    n = reflector.normal(u)
    d_to_camera = torch.nn.functional.normalize(c - x, dim=-1)
    r = 2.0 * (n * d_to_camera).sum(dim=-1, keepdim=True) * n - d_to_camera
    r = torch.nn.functional.normalize(r, dim=-1)

    h, w = camera.height, camera.width
    env = torch.tensor(environment_rgb, device=device, dtype=dtype)
    image = env.expand(h, w, 3).clone()
    best_t = torch.full((h, w), float("inf"), device=device, dtype=dtype)
    best_surface = torch.full((h, w), -1, device=device, dtype=torch.int64)

    hits: list[tuple[Tensor, Tensor, Tensor]] = []
    for surf in surfaces:
        hits.append(intersect_plane_surface(x, r, surf))

    for i, (surf, (t, s, valid)) in enumerate(zip(surfaces, hits)):
        take = valid & mirror_mask & (t < best_t)
        if not bool(take.any()):
            continue
        rgb = surf.radiance(s, -r)
        image[take] = rgb[take]
        best_t = torch.where(take, t, best_t)
        best_surface = torch.where(take, torch.full_like(best_surface, i), best_surface)

    image = torch.where(mirror_mask[..., None], image, torch.zeros_like(image))
    return RayReferenceResult(
        image=image,
        mirror_mask=mirror_mask,
        mirror_u=u,
        first_hit_tau=best_t,
        first_hit_surface=best_surface,
    )
