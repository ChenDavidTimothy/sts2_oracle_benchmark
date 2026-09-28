import pytest
import torch

from raster import has_cupy_cuda, rasterize_ewa, rasterize_triangles


pytestmark = pytest.mark.skipif(not has_cupy_cuda(), reason="CuPy CUDA backend unavailable")


def _compare(a, b):
    ai, ad, aid, ast = a
    bi, bd, bid, bst = b
    assert torch.equal(torch.isfinite(ad), torch.isfinite(bd))
    assert torch.equal(aid, bid)
    assert torch.allclose(ad[torch.isfinite(ad)], bd[torch.isfinite(bd)], atol=1e-5, rtol=0)
    assert torch.allclose(ai, bi, atol=2e-5, rtol=0)
    assert ast["pixel_touches"] == bst["pixel_touches"] > 0
    assert ast["depth_only_pixel_touches"] == bst["depth_only_pixel_touches"] > 0


def test_ewa_primitive_buffers_match():
    d = "cuda"
    p = torch.tensor([[8.0, 8.0], [8.0, 8.0], [18.0, 15.0]], device=d)
    tau = torch.tensor([2.0, 1.0, 1.5], device=d)
    color = torch.tensor([[0.8, 0.2, 0.3], [0.0, 0.0, 0.0], [0.2, 0.8, 0.3]], device=d)
    cov = torch.eye(2, device=d).expand(3, 2, 2).clone() * 2
    grad = torch.zeros((3, 2), device=d)
    rad = torch.tensor([True, False, True], device=d)
    ids = torch.tensor([0, 1, 0], device=d, dtype=torch.int32)
    args = (p, tau, color, cov, grad, rad, 24, 24)
    _compare(rasterize_ewa(*args, surface_ids=ids, backend="torch", max_radius=8),
             rasterize_ewa(*args, surface_ids=ids, backend="cupy", max_radius=8))


def test_triangle_primitive_buffers_match():
    d = "cuda"
    p = torch.tensor([[[3.,3.],[12.,3.],[3.,12.]], [[15.,12.],[22.,12.],[15.,20.]]], device=d)
    tau = torch.tensor([[2.,2.,2.],[1.,1.,1.]], device=d)
    color = torch.tensor([[[.8,.2,.3]]*3, [[0.,0.,0.]]*3], device=d)
    rad = torch.tensor([True, False], device=d)
    ids = torch.tensor([0, 1], device=d, dtype=torch.int32)
    args = (p, tau, color, rad, 24, 24)
    _compare(rasterize_triangles(*args, surface_ids=ids, backend="torch"),
             rasterize_triangles(*args, surface_ids=ids, backend="cupy"))
