import pytest
import torch

from raster import has_cupy_cuda, rasterize_uv_triangles


@pytest.mark.parametrize("backend,device", [
    ("torch", "cpu"),
    pytest.param("torch", "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cupy", "cuda", marks=pytest.mark.skipif(not has_cupy_cuda(), reason="CuPy CUDA unavailable")),
])
def test_uv_raster_interpolates_source_coordinates_and_gradient(backend, device):
    p = torch.tensor([[[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]]], device=device)
    tau = torch.ones((1, 3), device=device)
    uv = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], device=device)
    rad = torch.tensor([True], device=device)
    ids = torch.tensor([3], device=device, dtype=torch.int32)
    depth, first, uv_img, grad, _ = rasterize_uv_triangles(
        p, tau, uv, rad, 7, 7, surface_ids=ids, backend=backend
    )
    assert depth[2, 3].item() == pytest.approx(1.0, abs=1e-6)
    assert first[2, 3].item() == 3
    assert uv_img[2, 3].tolist() == pytest.approx([0.5, 0.25], abs=1e-5)
    assert grad[2, 3].flatten().tolist() == pytest.approx([0.25, 0.0, 0.0, 0.25], abs=1e-5)


def test_uv_raster_uses_nearest_triangle_before_shading():
    p = torch.tensor([
        [[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]],
        [[1.0, 1.0], [5.0, 1.0], [1.0, 5.0]],
    ])
    tau = torch.tensor([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    uv = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8]],
    ])
    rad = torch.tensor([True, False])
    ids = torch.tensor([0, 1], dtype=torch.int32)
    depth, first, uv_img, _, stats = rasterize_uv_triangles(
        p, tau, uv, rad, 7, 7, surface_ids=ids, backend="torch"
    )
    assert depth[2, 3].item() == pytest.approx(1.0)
    assert first[2, 3].item() == 1
    assert uv_img[2, 3].tolist() == pytest.approx([0.5, 0.35], abs=1e-6)
    assert stats["depth_only_pixel_touches"] > 0
