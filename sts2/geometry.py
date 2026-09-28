from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor


@dataclass
class HeightFieldReflector:
    """Reflector chart X(u,v) = [u, v, f(u,v)]."""

    kind: Literal["plane", "quadratic", "ripple"] = "quadratic"
    patch_half_extent: float = 1.25
    kx: float = -0.18
    ky: float = -0.12
    ripple_amp: float = 0.05
    ripple_freq: float = 2.5

    def f_grad_hess(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, y = u[..., 0], u[..., 1]
        if self.kind == "plane":
            f = torch.zeros_like(x)
            grad = torch.zeros((*u.shape[:-1], 2), dtype=u.dtype, device=u.device)
            hess = torch.zeros((*u.shape[:-1], 2, 2), dtype=u.dtype, device=u.device)
            return f, grad, hess

        if self.kind == "quadratic":
            f = 0.5 * (self.kx * x * x + self.ky * y * y)
            grad = torch.stack((self.kx * x, self.ky * y), dim=-1)
            hess = torch.zeros((*u.shape[:-1], 2, 2), dtype=u.dtype, device=u.device)
            hess[..., 0, 0] = self.kx
            hess[..., 1, 1] = self.ky
            return f, grad, hess

        if self.kind == "ripple":
            w = self.ripple_freq
            a = self.ripple_amp
            sx, cx = torch.sin(w * x), torch.cos(w * x)
            sy, cy = torch.sin(w * y), torch.cos(w * y)
            f_quad = 0.5 * (self.kx * x * x + self.ky * y * y)
            f = f_quad + a * sx * cy
            fx = self.kx * x + a * w * cx * cy
            fy = self.ky * y - a * w * sx * sy
            grad = torch.stack((fx, fy), dim=-1)
            hess = torch.zeros((*u.shape[:-1], 2, 2), dtype=u.dtype, device=u.device)
            hess[..., 0, 0] = self.kx - a * w * w * sx * cy
            hess[..., 1, 1] = self.ky - a * w * w * sx * cy
            cross = -a * w * w * cx * sy
            hess[..., 0, 1] = cross
            hess[..., 1, 0] = cross
            return f, grad, hess

        raise ValueError(f"unknown reflector kind: {self.kind}")

    def eval(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        f, grad, hess = self.f_grad_hess(u)
        x = torch.stack((u[..., 0], u[..., 1], f), dim=-1)
        j = torch.zeros((*u.shape[:-1], 3, 2), dtype=u.dtype, device=u.device)
        j[..., 0, 0] = 1.0
        j[..., 1, 1] = 1.0
        j[..., 2, 0] = grad[..., 0]
        j[..., 2, 1] = grad[..., 1]
        return x, j, hess

    def normal(self, u: Tensor) -> Tensor:
        _, grad, _ = self.f_grad_hess(u)
        n = torch.stack((-grad[..., 0], -grad[..., 1], torch.ones_like(grad[..., 0])), dim=-1)
        return torch.nn.functional.normalize(n, dim=-1)

    def in_bounds(self, u: Tensor, margin: float = 0.0) -> Tensor:
        h = self.patch_half_extent + margin
        return (u[..., 0].abs() <= h) & (u[..., 1].abs() <= h)


@dataclass
class PlaneSurface:
    name: str
    center: tuple[float, float, float]
    axis_u: tuple[float, float, float]
    axis_v: tuple[float, float, float]
    role: Literal["radiance", "depth_only"] = "radiance"
    texture: Literal["checker", "sine", "constant"] = "checker"
    bandwidth: float = 6.0
    constant_rgb: tuple[float, float, float] = (0.75, 0.75, 0.75)

    def tensors(self, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        c = torch.tensor(self.center, device=device, dtype=dtype)
        au = torch.tensor(self.axis_u, device=device, dtype=dtype)
        av = torch.tensor(self.axis_v, device=device, dtype=dtype)
        return c, au, av

    def eval(self, s: Tensor) -> tuple[Tensor, Tensor]:
        c, au, av = self.tensors(device=s.device, dtype=s.dtype)
        y = c + s[..., 0:1] * au + s[..., 1:2] * av
        j = torch.stack((au, av), dim=-1)
        j = j.expand(*s.shape[:-1], 3, 2)
        return y, j

    def normal(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        _, au, av = self.tensors(device=device, dtype=dtype)
        n = torch.cross(au, av, dim=0)
        return torch.nn.functional.normalize(n, dim=0)

    def coordinates(self, p: Tensor) -> Tensor:
        c, au, av = self.tensors(device=p.device, dtype=p.dtype)
        du = p - c
        # The benchmark uses orthogonal chart axes.
        su = (du * au).sum(dim=-1) / (au.square().sum() + 1e-20)
        sv = (du * av).sum(dim=-1) / (av.square().sum() + 1e-20)
        return torch.stack((su, sv), dim=-1)

    def in_bounds(self, s: Tensor, eps: float = 1e-6) -> Tensor:
        return (s[..., 0].abs() <= 1.0 + eps) & (s[..., 1].abs() <= 1.0 + eps)

    def radiance(self, s: Tensor, direction: Tensor | None = None) -> Tensor:
        del direction  # Stage-A source field is Lambertian by construction.
        if self.role == "depth_only":
            return torch.zeros((*s.shape[:-1], 3), dtype=s.dtype, device=s.device)
        if self.texture == "constant":
            rgb = torch.tensor(self.constant_rgb, device=s.device, dtype=s.dtype)
            return rgb.expand(*s.shape[:-1], 3)
        if self.texture == "checker":
            qx = torch.floor((s[..., 0] + 1.0) * 0.5 * self.bandwidth)
            qy = torch.floor((s[..., 1] + 1.0) * 0.5 * self.bandwidth)
            v = ((qx + qy).remainder(2.0)).to(s.dtype)
            c0 = torch.tensor((0.08, 0.18, 0.95), device=s.device, dtype=s.dtype)
            c1 = torch.tensor((0.95, 0.82, 0.08), device=s.device, dtype=s.dtype)
            return c0 + v[..., None] * (c1 - c0)
        if self.texture == "sine":
            w = float(self.bandwidth)
            x = s[..., 0]
            y = s[..., 1]
            r = 0.5 + 0.5 * torch.sin(w * x + 0.3 * torch.cos(0.7 * w * y))
            g = 0.5 + 0.5 * torch.sin(0.83 * w * y + 0.8)
            b = 0.5 + 0.5 * torch.sin(0.61 * w * (x + y) - 0.4)
            return torch.stack((r, g, b), dim=-1).clamp(0.0, 1.0)
        raise ValueError(f"unknown texture: {self.texture}")


def make_source_grid(n: int, *, device: torch.device, dtype: torch.dtype, vertices: bool = False) -> Tensor:
    if vertices:
        q = torch.linspace(-1.0, 1.0, n + 1, device=device, dtype=dtype)
    else:
        step = 2.0 / n
        q = torch.linspace(-1.0 + 0.5 * step, 1.0 - 0.5 * step, n, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def grid_triangles(n_cells: int, *, device: torch.device) -> Tensor:
    """Two triangles per cell for an (n_cells+1)^2 row-major vertex grid."""
    ids = torch.arange((n_cells + 1) ** 2, device=device, dtype=torch.int64).reshape(n_cells + 1, n_cells + 1)
    a = ids[:-1, :-1].reshape(-1)
    b = ids[:-1, 1:].reshape(-1)
    c = ids[1:, :-1].reshape(-1)
    d = ids[1:, 1:].reshape(-1)
    t0 = torch.stack((a, b, d), dim=-1)
    t1 = torch.stack((a, d, c), dim=-1)
    return torch.cat((t0, t1), dim=0)
