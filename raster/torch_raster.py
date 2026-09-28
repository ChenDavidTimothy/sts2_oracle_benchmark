from __future__ import annotations

import torch

Tensor = torch.Tensor


def rasterize_ewa_torch(
    p: Tensor,
    tau: Tensor,
    color: Tensor,
    cov: Tensor,
    depth_grad: Tensor,
    radiance_mask: Tensor,
    height: int,
    width: int,
    mirror_mask: Tensor | None = None,
    environment_rgb: tuple[float, float, float] = (0.03, 0.04, 0.06),
    max_radius: int = 6,
    depth_eps: float = 1e-3,
    chi2: float = 9.0,
    surface_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    device, dtype = p.device, p.dtype
    n = p.shape[0]
    if n == 0:
        env = torch.tensor(environment_rgb, device=device, dtype=dtype)
        img = env.expand(height, width, 3).clone()
        if mirror_mask is not None:
            img = torch.where(mirror_mask[..., None], img, torch.zeros_like(img))
        return img, torch.full((height, width), float("inf"), device=device, dtype=dtype), torch.full((height, width), -1, device=device, dtype=torch.int32), {"pixel_touches": 0, "depth_only_pixel_touches": 0}

    r = int(max_radius)
    oy, ox = torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device),
        indexing="ij",
    )
    offsets = torch.stack((ox, oy), dim=-1).reshape(-1, 2)
    k = offsets.shape[0]
    centers = torch.round(p).to(torch.int64)
    pix = centers[:, None, :] + offsets[None, :, :]
    inside = (pix[..., 0] >= 0) & (pix[..., 0] < width) & (pix[..., 1] >= 0) & (pix[..., 1] < height)
    delta = pix.to(dtype) - p[:, None, :]
    inv_cov = torch.linalg.inv(cov + 1e-6 * torch.eye(2, device=device, dtype=dtype)[None])
    q = torch.einsum("nki,nij,nkj->nk", delta, inv_cov, delta)
    support = inside & (q <= chi2)
    depth = tau[:, None] + torch.einsum("ni,nki->nk", depth_grad, delta)
    support &= depth > 0
    idx = pix[..., 1] * width + pix[..., 0]

    flat_support = support.reshape(-1)
    idx_f = idx.reshape(-1)[flat_support]
    depth_f = depth.reshape(-1)[flat_support]
    q_f = q.reshape(-1)[flat_support]
    sample_id = torch.arange(n, device=device)[:, None].expand(n, k).reshape(-1)[flat_support]

    depth_min = torch.full((height * width,), float("inf"), device=device, dtype=dtype)
    if idx_f.numel() > 0:
        depth_min.scatter_reduce_(0, idx_f, depth_f, reduce="amin", include_self=True)

    first_id = torch.full((height * width,), torch.iinfo(torch.int32).max, device=device, dtype=torch.int32)
    if surface_ids is None:
        surface_ids = torch.zeros((n,), device=device, dtype=torch.int32)
    if idx_f.numel() > 0:
        closest = depth_f == depth_min[idx_f]
        first_id.scatter_reduce_(0, idx_f[closest], surface_ids.to(torch.int32)[sample_id[closest]], reduce="amin", include_self=True)

    weight_sum = torch.zeros((height * width,), device=device, dtype=dtype)
    rgb_sum = torch.zeros((height * width, 3), device=device, dtype=dtype)
    if idx_f.numel() > 0:
        visible = depth_f <= depth_min[idx_f] + depth_eps
        visible &= radiance_mask[sample_id]
        visible &= surface_ids[sample_id] == first_id[idx_f]
        if bool(visible.any()):
            ids = idx_f[visible]
            sid = sample_id[visible]
            w = torch.exp(-0.5 * q_f[visible])
            weight_sum.scatter_add_(0, ids, w)
            for ch in range(3):
                rgb_sum[:, ch].scatter_add_(0, ids, w * color[sid, ch])

    env = torch.tensor(environment_rgb, device=device, dtype=dtype)
    img = env.expand(height * width, 3).clone()
    has_depth = torch.isfinite(depth_min)
    has_color = weight_sum > 1e-8
    img[has_depth] = 0.0  # depth-only first hits are black in Stage A2
    img[has_color] = rgb_sum[has_color] / weight_sum[has_color, None]
    img = img.reshape(height, width, 3)
    depth_img = depth_min.reshape(height, width)
    if mirror_mask is not None:
        img = torch.where(mirror_mask[..., None], img, torch.zeros_like(img))
        depth_img = torch.where(mirror_mask, depth_img, torch.full_like(depth_img, float("inf")))
    first_id = torch.where(first_id == torch.iinfo(torch.int32).max, -1, first_id).reshape(height, width)
    if mirror_mask is not None:
        first_id = torch.where(mirror_mask, first_id, -1)
    return img, depth_img, first_id, {"pixel_touches": int(idx_f.numel()), "depth_only_pixel_touches": int((~radiance_mask[sample_id]).sum().item())}


def _triangle_barycentric(tri: Tensor, pixels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # tri: T x 3 x 2, pixels: P x 2 -> T x P barycentric
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    v0 = b - a
    v1 = c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    safe = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
    v2 = pixels[None, :, :] - a[:, None, :]
    u = (v2[..., 0] * v1[:, None, 1] - v1[:, None, 0] * v2[..., 1]) / safe[:, None]
    v = (v0[:, None, 0] * v2[..., 1] - v2[..., 0] * v0[:, None, 1]) / safe[:, None]
    w = 1.0 - u - v
    inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6) & (den.abs()[:, None] > 1e-10)
    return w, u, v, inside


def rasterize_triangles_torch(
    tri_p: Tensor,
    tri_tau: Tensor,
    tri_color: Tensor,
    tri_radiance_mask: Tensor,
    height: int,
    width: int,
    mirror_mask: Tensor | None = None,
    environment_rgb: tuple[float, float, float] = (0.03, 0.04, 0.06),
    chunk: int = 64,
    depth_eps: float = 1e-3,
    surface_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    device, dtype = tri_p.device, tri_p.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    npix = pixels.shape[0]
    depth = torch.full((npix,), float("inf"), device=device, dtype=dtype)
    color = torch.zeros((npix, 3), device=device, dtype=dtype)
    hit_rad = torch.zeros((npix,), device=device, dtype=torch.bool)
    first_id = torch.full((npix,), -1, device=device, dtype=torch.int32)
    if surface_ids is None:
        surface_ids = torch.zeros((tri_p.shape[0],), device=device, dtype=torch.int32)
    touches = 0
    depth_only_touches = 0
    for start in range(0, tri_p.shape[0], chunk):
        end = min(start + chunk, tri_p.shape[0])
        tp = tri_p[start:end]
        tt = tri_tau[start:end]
        tc = tri_color[start:end]
        tr = tri_radiance_mask[start:end]
        w0, w1, w2, inside = _triangle_barycentric(tp, pixels)
        z = w0 * tt[:, 0:1] + w1 * tt[:, 1:2] + w2 * tt[:, 2:3]
        z = torch.where(inside & (z > 0), z, torch.full_like(z, float("inf")))
        covered = inside & (z > 0)
        touches += int(covered.sum())
        depth_only_touches += int((covered & ~tr[:,None]).sum())
        zmin, arg = z.min(dim=0)
        better = zmin < depth
        if not bool(better.any()):
            continue
        pix_ids = torch.arange(npix, device=device)[better]
        targ = arg[better]
        wb0 = w0[targ, pix_ids]
        wb1 = w1[targ, pix_ids]
        wb2 = w2[targ, pix_ids]
        cc = wb0[:, None] * tc[targ, 0] + wb1[:, None] * tc[targ, 1] + wb2[:, None] * tc[targ, 2]
        depth[better] = zmin[better]
        color[better] = cc
        hit_rad[better] = tr[targ]
        first_id[better] = surface_ids[start:end][targ].to(torch.int32)

    env = torch.tensor(environment_rgb, device=device, dtype=dtype)
    img = env.expand(npix, 3).clone()
    finite = torch.isfinite(depth)
    img[finite] = 0.0
    img[finite & hit_rad] = color[finite & hit_rad]
    img = img.reshape(height, width, 3)
    depth_img = depth.reshape(height, width)
    if mirror_mask is not None:
        img = torch.where(mirror_mask[..., None], img, torch.zeros_like(img))
        depth_img = torch.where(mirror_mask, depth_img, torch.full_like(depth_img, float("inf")))
        first_id = torch.where(mirror_mask.reshape(-1), first_id, -1)
    return img, depth_img, first_id.reshape(height, width), {"pixel_touches": touches, "depth_only_pixel_touches": depth_only_touches}
