from __future__ import annotations

import math
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def masked_mse(a: Tensor, b: Tensor, mask: Tensor | None = None) -> Tensor:
    e = (a - b).square().mean(dim=-1)
    if mask is None:
        return e.mean()
    w = mask.to(e.dtype)
    return (e * w).sum() / w.sum().clamp_min(1.0)


def psnr(a: Tensor, b: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None and not bool(mask.any()):
        return float("nan")
    mse = float(masked_mse(a, b, mask))
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def radiance_hit_mask(first_hit_surface: Tensor, surface_roles: Tensor) -> Tensor:
    valid = (first_hit_surface >= 0) & (first_hit_surface < surface_roles.numel())
    return valid & surface_roles[first_hit_surface.long().clamp(0, surface_roles.numel()-1)].bool()


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) * 0.5
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    k = g[:, None] * g[None, :]
    return k


def ssim(a: Tensor, b: Tensor, mask: Tensor | None = None, window: int = 7, sigma: float = 1.5) -> float:
    # Images are HxWx3 in [0,1]. This is a compact RGB SSIM for benchmark comparisons.
    b = b.to(device=a.device, dtype=a.dtype)
    x = a.permute(2, 0, 1).unsqueeze(0)
    y = b.permute(2, 0, 1).unsqueeze(0)
    k = _gaussian_kernel(window, sigma, a.device, a.dtype).expand(3, 1, window, window)
    pad = window // 2
    mu_x = F.conv2d(x, k, padding=pad, groups=3)
    mu_y = F.conv2d(y, k, padding=pad, groups=3)
    ex2 = F.conv2d(x * x, k, padding=pad, groups=3)
    ey2 = F.conv2d(y * y, k, padding=pad, groups=3)
    exy = F.conv2d(x * y, k, padding=pad, groups=3)
    var_x = (ex2 - mu_x * mu_x).clamp_min(0)
    var_y = (ey2 - mu_y * mu_y).clamp_min(0)
    cov = exy - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
    m = m.mean(dim=1).squeeze(0)
    if mask is None:
        return float(m.mean())
    w = mask.to(m.dtype)
    return float((m * w).sum() / w.sum().clamp_min(1.0))


def depth_agreement(pred_depth: Tensor, ref_depth: Tensor, mask: Tensor, tol: float = 1e-2) -> float:
    ref_hits = torch.isfinite(ref_depth) & mask
    if not bool(ref_hits.any()):
        return float("nan")
    matched = torch.isfinite(pred_depth) & ref_hits
    correct = torch.zeros_like(ref_hits)
    correct[matched] = (pred_depth[matched] - ref_depth[matched]).abs() <= tol
    return float(correct[ref_hits].float().mean())


def visibility_metrics(
    pred_depth: Tensor, ref_depth: Tensor, pred_surface: Tensor,
    ref_surface: Tensor, mask: Tensor, tol: float = 1e-2,
    surface_roles: Tensor | None = None,
) -> dict[str, float | int]:
    pred = torch.isfinite(pred_depth) & mask
    ref = torch.isfinite(ref_depth) & mask
    matched = pred & ref
    tp = int(matched.sum().item())
    pred_count = int(pred.sum().item())
    ref_count = int(ref.sum().item())
    precision = tp / pred_count if pred_count else (1.0 if not ref_count else 0.0)
    recall = tp / ref_count if ref_count else 1.0
    errors = (pred_depth[matched] - ref_depth[matched]).abs()
    id_correct = int((pred_surface[matched] == ref_surface[matched]).sum().item())
    out = {
        "reflected_hit_count": ref_count,
        "predicted_hit_count": pred_count,
        "matched_hit_count": tp,
        "hit_precision": precision,
        "hit_recall": recall,
        "hit_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "matched_depth_rmse": float(torch.sqrt((errors * errors).mean())) if tp else float("nan"),
        "matched_depth_max_error": float(errors.max()) if tp else float("nan"),
        "matched_depth_within_tol": float((errors <= tol).float().mean()) if tp else float("nan"),
        "first_hit_surface_agreement": id_correct / tp if tp else float("nan"),
        "first_hit_surface_recall": id_correct / ref_count if ref_count else 1.0,
    }
    if surface_roles is not None:
        pred_ids = pred_surface[matched].long()
        ref_ids = ref_surface[matched].long()
        valid_ids = (pred_ids >= 0) & (pred_ids < surface_roles.numel()) & (ref_ids >= 0) & (ref_ids < surface_roles.numel())
        role_correct = int((valid_ids & (surface_roles[pred_ids.clamp(0, surface_roles.numel()-1)] == surface_roles[ref_ids.clamp(0, surface_roles.numel()-1)])).sum().item())
        out["first_hit_role_agreement"] = role_correct / tp if tp else float("nan")
        out["first_hit_role_recall"] = role_correct / ref_count if ref_count else 1.0
    return out


def maybe_lpips(a: Tensor, b: Tensor, enabled: bool = False) -> float | None:
    if not enabled:
        return None
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    net = lpips.LPIPS(net="alex").to(a.device)
    xa = a.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    xb = b.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        return float(net(xa, xb).item())
