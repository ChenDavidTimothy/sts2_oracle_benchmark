import torch

from sts2.geometry import HeightFieldReflector, PlaneSurface
from sts2.transport import solve_specular


def analytic_plane_root(y, c):
    # Reflect source across z=0, intersect line from camera to virtual source with z=0.
    ym = y.clone(); ym[..., 2] *= -1
    q = ym - c
    t = -c[2] / q[..., 2]
    x = c + t[..., None] * q
    return x[..., :2]


def test_planar_root_matches_virtual_source():
    dtype = torch.float64
    refl = HeightFieldReflector(kind="plane", patch_half_extent=2.0)
    surf = PlaneSurface("src", (0.7, 0.0, 2.0), (0.5, 0, 0), (0, 0.5, 0))
    s = torch.tensor([[0.2, -0.3], [-0.1, 0.5]], dtype=dtype)
    c = torch.tensor([-0.8, 0.1, 2.7], dtype=dtype)
    y, _ = surf.eval(s)
    expected = analytic_plane_root(y, c)
    got = solve_specular(refl, surf, s, c).u
    assert torch.max(torch.abs(expected - got)) < 1e-9
