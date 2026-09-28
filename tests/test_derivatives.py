import torch

from sts2.camera import PinholeCamera
from sts2.geometry import HeightFieldReflector, PlaneSurface
from sts2.transport import specular_terms, solve_specular, differentials


def finite_jac(fn, x, eps=1e-6):
    y0 = fn(x)
    out = torch.zeros((*y0.shape, x.shape[-1]), dtype=x.dtype)
    for j in range(x.shape[-1]):
        d = torch.zeros_like(x)
        d[..., j] = eps
        out[..., j] = (fn(x + d) - fn(x - d)) / (2 * eps)
    return out


def test_specular_derivatives():
    dtype = torch.float64
    refl = HeightFieldReflector(kind="ripple", kx=-0.18, ky=-0.09, ripple_amp=0.025, ripple_freq=1.8)
    surf = PlaneSurface("src", (0.75, 0.1, 2.0), (0.7, 0, 0), (0, 0.6, 0))
    cam = PinholeCamera(center=(-0.7, 0.15, 2.6), width=64, height=64)
    s = torch.tensor([[0.13, -0.21]], dtype=dtype)
    c = torch.tensor(cam.center, dtype=dtype)
    sol = solve_specular(refl, surf, s, c)
    assert bool(sol.valid[0])
    u = sol.u
    q = specular_terms(refl, surf, u, s, c)
    fd_u = finite_jac(lambda uu: specular_terms(refl, surf, uu, s, c)["F"], u)
    fd_s = finite_jac(lambda ss: specular_terms(refl, surf, u, ss, c)["F"], s)
    fd_c = finite_jac(lambda cc: specular_terms(refl, surf, u, s, cc)["F"], c)
    assert torch.max(torch.abs(fd_u - q["F_u"])) < 2e-7
    assert torch.max(torch.abs(fd_s - q["F_s"])) < 2e-7
    assert torch.max(torch.abs(fd_c - q["F_c"])) < 2e-7

    diff = differentials(refl, surf, cam, s, u)
    # Check du/ds through fully re-solved roots.
    eps = 1e-5
    cols = []
    for j in range(2):
        d = torch.zeros_like(s); d[:, j] = eps
        up = solve_specular(refl, surf, s + d, c, init_u=u).u
        um = solve_specular(refl, surf, s - d, c, init_u=u).u
        cols.append(((up - um) / (2 * eps)).squeeze(0))
    fd_jus = torch.stack(cols, dim=-1).unsqueeze(0)
    assert torch.max(torch.abs(fd_jus - diff.j_us)) < 2e-5

    cam_cols = []
    for j in range(3):
        d = torch.zeros_like(c); d[j] = eps
        up = solve_specular(refl, surf, s, c + d, init_u=u).u
        um = solve_specular(refl, surf, s, c - d, init_u=u).u
        cam_cols.append(((up - um) / (2 * eps)).squeeze(0))
    fd_juc = torch.stack(cam_cols, dim=-1).unsqueeze(0)
    assert torch.max(torch.abs(fd_juc - diff.j_uc)) < 2e-5

    tau_cols = []
    phi_cols = []
    for j in range(2):
        d = torch.zeros_like(s); d[:, j] = eps
        sp, sm = s + d, s - d
        up = solve_specular(refl, surf, sp, c, init_u=u).u
        um = solve_specular(refl, surf, sm, c, init_u=u).u
        xp, _, _ = refl.eval(up)
        xm, _, _ = refl.eval(um)
        yp, _ = surf.eval(sp)
        ym, _ = surf.eval(sm)
        tau_cols.append(((torch.linalg.vector_norm(yp-xp, dim=-1) - torch.linalg.vector_norm(ym-xm, dim=-1)) / (2*eps)).squeeze(0))
        pp, _ = cam.project(xp)
        pm, _ = cam.project(xm)
        phi_cols.append(((pp-pm)/(2*eps)).squeeze(0))
    fd_tau_s = torch.stack(tau_cols, dim=-1)
    fd_phi = torch.stack(phi_cols, dim=-1).unsqueeze(0)
    assert torch.max(torch.abs(fd_tau_s - diff.j_tau_s)) < 2e-5
    assert torch.max(torch.abs(fd_phi - diff.j_phi)) < 2e-4
