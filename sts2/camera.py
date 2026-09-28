from __future__ import annotations

from dataclasses import dataclass
import math
import torch

Tensor = torch.Tensor


@dataclass
class PinholeCamera:
    center: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    width: int = 192
    height: int = 192
    fov_y_deg: float = 42.0

    def matrices(self, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor, float, float]:
        c = torch.tensor(self.center, device=device, dtype=dtype)
        t = torch.tensor(self.target, device=device, dtype=dtype)
        up0 = torch.tensor(self.up, device=device, dtype=dtype)
        fwd = torch.nn.functional.normalize(t - c, dim=0)
        right = torch.nn.functional.normalize(torch.cross(fwd, up0, dim=0), dim=0)
        up = torch.cross(right, fwd, dim=0)
        r = torch.stack((right, up, fwd), dim=0)
        fy = 0.5 * self.height / math.tan(0.5 * math.radians(self.fov_y_deg))
        fx = fy
        return c, r, fwd, fx, fy

    def project(self, p: Tensor) -> tuple[Tensor, Tensor]:
        c, r, _, fx, fy = self.matrices(device=p.device, dtype=p.dtype)
        pc = torch.einsum("ij,...j->...i", r, p - c)
        z = pc[..., 2]
        eps = torch.finfo(p.dtype).eps * 16
        zsafe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
        px = fx * pc[..., 0] / zsafe + (self.width - 1) * 0.5
        py = -fy * pc[..., 1] / zsafe + (self.height - 1) * 0.5
        return torch.stack((px, py), dim=-1), z

    def jacobian(self, p: Tensor) -> Tensor:
        c, r, _, fx, fy = self.matrices(device=p.device, dtype=p.dtype)
        pc = torch.einsum("ij,...j->...i", r, p - c)
        x, y, z = pc.unbind(dim=-1)
        eps = torch.finfo(p.dtype).eps * 16
        z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
        jc = torch.zeros((*p.shape[:-1], 2, 3), device=p.device, dtype=p.dtype)
        jc[..., 0, 0] = fx / z
        jc[..., 0, 2] = -fx * x / (z * z)
        jc[..., 1, 1] = -fy / z
        jc[..., 1, 2] = fy * y / (z * z)
        return torch.einsum("...ij,jk->...ik", jc, r)

    def rays(self, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        c, r, _, fx, fy = self.matrices(device=device, dtype=dtype)
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=device, dtype=dtype),
            torch.arange(self.width, device=device, dtype=dtype),
            indexing="ij",
        )
        xc = (xx - (self.width - 1) * 0.5) / fx
        yc = -(yy - (self.height - 1) * 0.5) / fy
        zc = torch.ones_like(xc)
        dcam = torch.stack((xc, yc, zc), dim=-1)
        dcam = torch.nn.functional.normalize(dcam, dim=-1)
        dworld = torch.einsum("ij,...j->...i", r.T, dcam)
        o = c.expand_as(dworld)
        return o, dworld
