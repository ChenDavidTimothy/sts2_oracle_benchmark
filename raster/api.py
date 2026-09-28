from __future__ import annotations

import torch
from .torch_raster import rasterize_ewa_torch, rasterize_triangles_torch
try:
    from .cupy_raster import available as _cupy_available, rasterize_ewa_cupy, rasterize_triangles_cupy
except Exception:  # pragma: no cover
    _cupy_available = lambda: False
    rasterize_ewa_cupy = None
    rasterize_triangles_cupy = None


def has_cupy_cuda() -> bool:
    return bool(_cupy_available())


def rasterize_ewa(*args, backend: str = "auto", **kwargs):
    p = args[0] if args else kwargs["p"]
    use_cupy = backend == "cupy" or (backend == "auto" and p.device.type == "cuda" and has_cupy_cuda())
    if use_cupy:
        return rasterize_ewa_cupy(*args, **kwargs)
    return rasterize_ewa_torch(*args, **kwargs)


def rasterize_triangles(*args, backend: str = "auto", **kwargs):
    tri_p = args[0] if args else kwargs["tri_p"]
    use_cupy = backend == "cupy" or (backend == "auto" and tri_p.device.type == "cuda" and has_cupy_cuda())
    if use_cupy:
        return rasterize_triangles_cupy(*args, **kwargs)
    return rasterize_triangles_torch(*args, **kwargs)
