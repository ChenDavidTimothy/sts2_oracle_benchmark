from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .geometry import HeightFieldReflector, PlaneSurface
from .camera import PinholeCamera

Tensor = torch.Tensor


@dataclass
class SolveResult:
    u: Tensor
    converged: Tensor
    valid: Tensor
    residual: Tensor
    iterations: int


@dataclass
class DifferentialResult:
    u: Tensor
    x: Tensor
    y: Tensor
    ell: Tensor
    d: Tensor
    f_u: Tensor
    f_s: Tensor
    f_c: Tensor
    j_us: Tensor
    j_uc: Tensor
    j_phi: Tensor
    tau: Tensor
    j_tau_s: Tensor
    depth_grad: Tensor
    kappa_fu: Tensor
    sigma_min_jphi: Tensor


def _outer(a: Tensor, b: Tensor) -> Tensor:
    return a[..., :, None] * b[..., None, :]


def specular_terms(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    u: Tensor,
    s: Tensor,
    camera_center: Tensor,
) -> dict[str, Tensor]:
    x, jx, hess_f = reflector.eval(u)
    y, jy = surface.eval(s)
    ry_vec = y - x
    rc_vec = camera_center - x
    r_y = torch.linalg.vector_norm(ry_vec, dim=-1).clamp_min(1e-12)
    r_c = torch.linalg.vector_norm(rc_vec, dim=-1).clamp_min(1e-12)
    ell = ry_vec / r_y[..., None]
    d = rc_vec / r_c[..., None]
    f = torch.einsum("...ji,...j->...i", jx, ell + d)

    eye3 = torch.eye(3, device=u.device, dtype=u.dtype)
    p_ell = eye3 - _outer(ell, ell)
    p_d = eye3 - _outer(d, d)
    a = p_ell / r_y[..., None, None] + p_d / r_c[..., None, None]
    jtaj = torch.einsum("...ki,...kl,...lj->...ij", jx, a, jx)
    # For a height field X=[u,v,f(u,v)], the contracted chart Hessian is v_z Hessian(f).
    h_contract = (ell[..., 2] + d[..., 2])[..., None, None] * hess_f
    f_u = h_contract - jtaj

    f_s = torch.einsum(
        "...ki,...kl,...lj->...ij",
        jx,
        p_ell / r_y[..., None, None],
        jy,
    )
    f_c = torch.einsum(
        "...ki,...kl->...il",
        jx,
        p_d / r_c[..., None, None],
    )
    return {
        "x": x,
        "jx": jx,
        "y": y,
        "jy": jy,
        "r_y": r_y,
        "r_c": r_c,
        "ell": ell,
        "d": d,
        "F": f,
        "F_u": f_u,
        "F_s": f_s,
        "F_c": f_c,
    }


def _solve_linear(a: Tensor, b: Tensor) -> Tensor:
    """Solve a small batched linear system with a least-squares fallback."""
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        return torch.linalg.lstsq(a, b).solution


def newton_correct(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    s: Tensor,
    camera_center: Tensor,
    u: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply exactly one Newton correction to the stationary-path constraint.

    Returns (u_next, du). No line search or patch clamp is applied: this is the
    fixed production correction used by STS-2.1. Callers must run the physical
    validity / conditioning gates afterwards.
    """
    q = specular_terms(reflector, surface, u, s, camera_center)
    eye2 = torch.eye(2, device=u.device, dtype=u.dtype).expand_as(q["F_u"])
    du = _solve_linear(q["F_u"] + 1e-10 * eye2, -q["F"][..., None]).squeeze(-1)
    return u + du, du


def planar_initial_guess(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    s: Tensor,
    camera_center: Tensor,
) -> Tensor:
    # Tangent plane at chart origin. For the height-field benchmark this is enough to seed mild curvature.
    origin_u = torch.zeros((1, 2), device=s.device, dtype=s.dtype)
    p0, _, _ = reflector.eval(origin_u)
    p0 = p0[0]
    n0 = reflector.normal(origin_u)[0]
    y, _ = surface.eval(s)
    dist = ((y - p0) * n0).sum(dim=-1, keepdim=True)
    y_mirror = y - 2.0 * dist * n0
    q = y_mirror - camera_center
    denom = (q * n0).sum(dim=-1)
    numer = ((p0 - camera_center) * n0).sum()
    safe = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    t = numer / safe
    hit = camera_center + t[..., None] * q
    return hit[..., :2].clamp(-reflector.patch_half_extent, reflector.patch_half_extent)


def solve_specular(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    s: Tensor,
    camera_center: Tensor,
    init_u: Tensor | None = None,
    max_iter: int = 20,
    tol: float = 1e-8,
    damping_floor: float = 0.0625,
) -> SolveResult:
    if init_u is None:
        u = planar_initial_guess(reflector, surface, s, camera_center)
    else:
        u = init_u.clone()

    eye2 = torch.eye(2, device=u.device, dtype=u.dtype)
    it_used = 0
    for it in range(max_iter):
        it_used = it + 1
        q = specular_terms(reflector, surface, u, s, camera_center)
        f = q["F"]
        fn = torch.linalg.vector_norm(f, dim=-1)
        if bool((fn < tol).all()):
            break
        fu = q["F_u"]
        # Small Tikhonov term only for numerical solving; diagnostics use the unregularized matrix.
        reg = 1e-10 * eye2.expand_as(fu)
        du = _solve_linear(fu + reg, -f[..., None]).squeeze(-1)

        # Vectorized backtracking over four damping candidates.
        alphas = torch.tensor([1.0, 0.5, 0.25, 0.125, damping_floor], device=u.device, dtype=u.dtype)
        cand = u[:, None, :] + alphas[None, :, None] * du[:, None, :]
        cand = cand.clamp(-reflector.patch_half_extent, reflector.patch_half_extent)
        n, a, _ = cand.shape
        srep = s[:, None, :].expand(n, a, 2).reshape(-1, 2)
        frep = specular_terms(reflector, surface, cand.reshape(-1, 2), srep, camera_center)["F"]
        norms = torch.linalg.vector_norm(frep, dim=-1).reshape(n, a)
        best = norms.argmin(dim=1)
        u = cand[torch.arange(n, device=u.device), best]

    q = specular_terms(reflector, surface, u, s, camera_center)
    residual = torch.linalg.vector_norm(q["F"], dim=-1)
    converged = residual < max(tol * 10.0, 1e-7)
    n = reflector.normal(u)
    same_side = ((n * q["ell"]).sum(dim=-1) > 0) & ((n * q["d"]).sum(dim=-1) > 0)
    valid = converged & reflector.in_bounds(u) & same_side
    return SolveResult(u=u, converged=converged, valid=valid, residual=residual, iterations=it_used)


def differentials(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    s: Tensor,
    u: Tensor,
) -> DifferentialResult:
    c, _, _, _, _ = camera.matrices(device=s.device, dtype=s.dtype)
    q = specular_terms(reflector, surface, u, s, c)
    fu, fs, fc = q["F_u"], q["F_s"], q["F_c"]
    eye2 = torch.eye(2, device=s.device, dtype=s.dtype).expand_as(fu)
    fu_reg = fu + 1e-10 * eye2
    j_us = -_solve_linear(fu_reg, fs)
    j_uc = -_solve_linear(fu_reg, fc)
    j_pi = camera.jacobian(q["x"])
    j_phi = torch.einsum("...ij,...jk,...kl->...il", j_pi, q["jx"], j_us)
    tau = q["r_y"]
    j_tau_s = torch.einsum("...i,...ij->...j", q["ell"], q["jy"] - torch.einsum("...ij,...jk->...ik", q["jx"], j_us))

    # Screen-space depth gradient only where J_phi is well conditioned.
    svals = torch.linalg.svdvals(j_phi)
    sigma_min = svals[..., -1]
    inv_jphi = torch.linalg.pinv(j_phi, rcond=1e-8)
    depth_grad = torch.einsum("...i,...ij->...j", j_tau_s, inv_jphi)
    svals_fu = torch.linalg.svdvals(fu)
    kappa_fu = svals_fu[..., 0] / svals_fu[..., -1].clamp_min(1e-12)
    return DifferentialResult(
        u=u,
        x=q["x"],
        y=q["y"],
        ell=q["ell"],
        d=q["d"],
        f_u=fu,
        f_s=fs,
        f_c=fc,
        j_us=j_us,
        j_uc=j_uc,
        j_phi=j_phi,
        tau=tau,
        j_tau_s=j_tau_s,
        depth_grad=depth_grad,
        kappa_fu=kappa_fu,
        sigma_min_jphi=sigma_min,
    )


def multistart_roots(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    s: Tensor,
    camera_center: Tensor,
    seeds_per_axis: int = 5,
    cluster_eps: float = 2e-3,
) -> list[list[Tensor]]:
    """Slow diagnostic root discovery for Stage A3. Intended for tens of anchors, not rendering."""
    q = torch.linspace(-reflector.patch_half_extent, reflector.patch_half_extent, seeds_per_axis, device=s.device, dtype=s.dtype)
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    seeds = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    out: list[list[Tensor]] = []
    for i in range(s.shape[0]):
        si = s[i : i + 1].expand(seeds.shape[0], 2)
        sol = solve_specular(reflector, surface, si, camera_center, init_u=seeds, max_iter=30)
        roots: list[Tensor] = []
        for u, ok in zip(sol.u, sol.valid):
            if not bool(ok):
                continue
            if all(float(torch.linalg.vector_norm(u - r)) > cluster_eps for r in roots):
                roots.append(u.detach().clone())
        out.append(roots)
    return out
