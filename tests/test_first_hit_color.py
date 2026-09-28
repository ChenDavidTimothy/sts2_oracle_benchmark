import pytest
import torch

from raster import has_cupy_cuda, rasterize_ewa, rasterize_triangles


@pytest.mark.parametrize("backend,device", [
    ("torch", "cpu"),
    pytest.param("torch", "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cupy", "cuda", marks=pytest.mark.skipif(not has_cupy_cuda(), reason="CuPy CUDA unavailable")),
])
@pytest.mark.parametrize("raster", ["ewa", "triangles"])
def test_near_depth_occluder_blocks_other_surface_color(backend, device, raster):
    # The red source lies within depth_eps behind the black depth-only winner.
    # Same-surface samples may still blend within that tolerance.
    dtype = torch.float32
    tau = torch.tensor([1.0, 1.0005, 1.0007], device=device, dtype=dtype)
    color = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], device=device, dtype=dtype)
    rad = torch.tensor([False, True, True], device=device)
    ids = torch.tensor([1, 0, 1], device=device, dtype=torch.int32)
    if raster == "ewa":
        p = torch.tensor([[4., 4.]] * 3, device=device, dtype=dtype)
        cov = torch.eye(2, device=device, dtype=dtype).expand(3, 2, 2).clone()
        grad = torch.zeros((3, 2), device=device, dtype=dtype)
        image, depth, first_id, _ = rasterize_ewa(
            p, tau, color, cov, grad, rad, 9, 9,
            surface_ids=ids, depth_eps=1e-3, backend=backend,
        )
    else:
        p = torch.tensor([[[2., 2.], [6., 2.], [4., 6.]]] * 2, device=device, dtype=dtype)
        tri_tau = tau[:2, None].expand(2, 3).contiguous()
        tri_color = color[:2, None, :].expand(2, 3, 3).contiguous()
        image, depth, first_id, _ = rasterize_triangles(
            p, tri_tau, tri_color, rad[:2], 9, 9,
            surface_ids=ids[:2], depth_eps=1e-3, backend=backend,
        )
    assert depth[4, 4].item() == pytest.approx(1.0)
    assert first_id[4, 4].item() == 1
    expected = [0., 1., 0.] if raster == "ewa" else [0., 0., 0.]
    assert image[4, 4].tolist() == pytest.approx(expected, abs=1e-6)


@pytest.mark.skipif(not has_cupy_cuda(), reason="CuPy CUDA unavailable")
@pytest.mark.parametrize("raster", ["ewa", "triangles"])
def test_near_depth_backend_parity(raster):
    # The shared first-hit model must make both backends return black at the
    # occluded pixel, regardless of primitive order.
    d = "cuda"
    ids = torch.tensor([1, 0], device=d, dtype=torch.int32)
    rad = torch.tensor([False, True], device=d)
    tau = torch.tensor([1.0, 1.0005], device=d)
    color = torch.tensor([[0., 0., 0.], [1., 0., 0.]], device=d)
    def run(backend):
        if raster == "ewa":
            p = torch.tensor([[4., 4.], [4., 4.]], device=d)
            cov = torch.eye(2, device=d).expand(2, 2, 2).clone()
            grad = torch.zeros((2, 2), device=d)
            return rasterize_ewa(p, tau, color, cov, grad, rad, 9, 9, surface_ids=ids, backend=backend)
        p = torch.tensor([[[2., 2.], [6., 2.], [4., 6.]]] * 2, device=d)
        return rasterize_triangles(p, tau[:,None].expand(2,3).contiguous(), color[:,None,:].expand(2,3,3).contiguous(), rad, 9, 9, surface_ids=ids, backend=backend)
    a, b = run("torch"), run("cupy")
    assert torch.equal(a[2], b[2])
    assert torch.allclose(a[0], b[0], atol=1e-6, rtol=0)
    assert a[0][4, 4].tolist() == pytest.approx([0., 0., 0.], abs=1e-6)
