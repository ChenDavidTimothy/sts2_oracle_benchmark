import torch

from sts2.camera import PinholeCamera
from sts2.geometry import HeightFieldReflector, PlaneSurface
from sts2.transport import solve_specular, differentials


def test_first_order_transport_is_second_order_accurate():
    dtype = torch.float64
    refl = HeightFieldReflector(kind="quadratic", kx=-0.22, ky=-0.11, patch_half_extent=1.5)
    surf = PlaneSurface("src", (0.85, 0.0, 2.1), (0.75, 0, 0), (0, 0.65, 0))
    cam = PinholeCamera(center=(-0.75, 0.15, 2.7), width=64, height=64)
    c = torch.tensor(cam.center, dtype=dtype)
    s0 = torch.tensor([[0.05, -0.08]], dtype=dtype)
    u0 = solve_specular(refl, surf, s0, c).u
    j = differentials(refl, surf, cam, s0, u0).j_us[0]
    direction = torch.tensor([0.8, -0.6], dtype=dtype)
    errors = []
    hs = [0.01, 0.02, 0.04]
    for h in hs:
        s = s0 + h * direction
        exact = solve_specular(refl, surf, s, c, init_u=u0).u[0]
        pred = u0[0] + j @ (h * direction)
        errors.append(float(torch.linalg.vector_norm(exact - pred)))
    assert errors[1] / errors[0] > 3.2
    assert errors[2] / errors[1] > 3.0
