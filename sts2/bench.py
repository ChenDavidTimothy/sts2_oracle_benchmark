from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import os
import platform
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .metrics import psnr, ssim, depth_agreement, visibility_metrics, radiance_hit_mask, maybe_lpips
from .ray_reference import render_reference
from .render import render_ewa, render_triangles, render_uv_triangles
from .adaptive import AdaptiveTransportConfig
from .scenes import scene_from_config, expand_sweep
from .utils import save_image, save_json, seed_all, timed
from raster import has_cupy_cuda


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _system_info() -> dict:
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        out["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
    try:
        out["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        out["git_commit"] = None
    return out


def _run_one(cfg: dict, out_dir: Path, device: torch.device) -> list[dict]:
    scene = scene_from_config(cfg["scene"])
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    reference, ref_ms, _ = timed(
        lambda: render_reference(
            scene.reflector, scene.camera, scene.surfaces,
            device=device, dtype=dtype, environment_rgb=scene.environment_rgb,
        ),
        warmup=cfg["benchmark"].get("warmup", 1),
        repeats=cfg["benchmark"].get("repeats", 3),
    )
    save_image(out_dir / "reference.png", reference.image)
    np.save(out_dir / "reference_depth.npy", reference.first_hit_tau.detach().cpu().numpy())
    np.save(out_dir / "reference_first_hit_surface.npy", reference.first_hit_surface.detach().cpu().numpy())
    rows: list[dict] = []
    rows.append({
        "method": "analytic_ray_reference",
        "frame_ms": ref_ms,
        "psnr_reflect": 99.0,
        "ssim_reflect": 1.0,
        "depth_agreement": 1.0,
        "pixel_touches": int(reference.mirror_mask.sum().item()),
        "exact_anchor_solves": 0,
    })

    methods = cfg["benchmark"].get("methods", ["exact_ewa", "tiled_ewa", "exact_tri", "tiled_tri"])
    n = int(cfg["benchmark"].get("source_resolution", 64))
    tiles = int(cfg["benchmark"].get("tiles_per_axis", 8))
    backend = cfg["benchmark"].get("raster_backend", "auto")
    lpips_enabled = bool(cfg["benchmark"].get("lpips", False))
    filter_taps = int(cfg["benchmark"].get("source_filter_taps", 4))
    adaptive_base = AdaptiveTransportConfig(**cfg["benchmark"].get("adaptive", {}))
    msaa = int(cfg["benchmark"].get("triangle_msaa", 1))
    if msaa not in (1, 4):
        raise ValueError("triangle_msaa must be 1 or 4")
    parity_gate = bool(cfg["benchmark"].get("require_backend_parity", False)) and device.type == "cuda"
    if parity_gate and not has_cupy_cuda():
        raise RuntimeError("backend parity requires CuPy")
    msaa_scene = None
    msaa_ref = None
    if msaa == 4:
        msaa_scene = replace(scene, camera=replace(scene.camera, width=2*scene.camera.width, height=2*scene.camera.height))
        msaa_ref = render_reference(msaa_scene.reflector, msaa_scene.camera, msaa_scene.surfaces, device=device, dtype=dtype, environment_rgb=scene.environment_rgb)
    roles = torch.tensor([int(s.role == "radiance") for s in scene.surfaces], device=device, dtype=torch.int32)
    ref_radiance = radiance_hit_mask(reference.first_hit_surface, roles)
    results = {}
    quality_rows = {}
    msaa_images = {}
    msaa_radiance_masks = {}

    def evaluate(method: str, chosen_backend: str | None = None, high_res: bool = False):
        active = msaa_scene if high_res else scene
        mask = msaa_ref.mirror_mask if high_res else reference.mirror_mask
        chosen_backend = chosen_backend or backend
        if method == "exact_ewa":
            return render_ewa(active.reflector, active.camera, active.surfaces, mask, n_samples=n, mode="exact", backend=chosen_backend, environment_rgb=scene.environment_rgb)
        if method == "tiled_ewa":
            return render_ewa(active.reflector, active.camera, active.surfaces, mask, n_samples=n, mode="tiled", tiles_per_axis=tiles, backend=chosen_backend, environment_rgb=scene.environment_rgb)
        if method == "exact_tri":
            return render_triangles(active.reflector, active.camera, active.surfaces, mask, n_cells=n, mode="exact", backend=chosen_backend, environment_rgb=scene.environment_rgb)
        if method == "tiled_tri":
            return render_triangles(active.reflector, active.camera, active.surfaces, mask, n_cells=n, mode="tiled", tiles_per_axis=tiles, backend=chosen_backend, environment_rgb=scene.environment_rgb)
        if method == "exact_tri_uv":
            return render_uv_triangles(
                active.reflector, active.camera, active.surfaces, mask,
                mode="exact", n_cells=n, backend=chosen_backend,
                environment_rgb=scene.environment_rgb, filter_taps=filter_taps,
            )
        if method == "adaptive_pred_tri_uv":
            acfg = replace(adaptive_base, newton_corrections=0)
            return render_uv_triangles(
                active.reflector, active.camera, active.surfaces, mask,
                mode="adaptive", adaptive_cfg=acfg, backend=chosen_backend,
                environment_rgb=scene.environment_rgb, filter_taps=filter_taps,
            )
        if method == "adaptive_newton_tri_uv":
            acfg = replace(adaptive_base, newton_corrections=1)
            return render_uv_triangles(
                active.reflector, active.camera, active.surfaces, mask,
                mode="adaptive", adaptive_cfg=acfg, backend=chosen_backend,
                environment_rgb=scene.environment_rgb, filter_taps=filter_taps,
            )
        raise ValueError(f"unknown method: {method}")

    for method in methods:
        result, ms, timing = timed(
            lambda m=method: evaluate(m),
            warmup=cfg["benchmark"].get("warmup", 1),
            repeats=cfg["benchmark"].get("repeats", 3),
        )
        save_image(out_dir / f"{method}.png", result.image)
        np.save(out_dir / f"{method}_depth.npy", result.depth.detach().cpu().numpy())
        np.save(out_dir / f"{method}_first_hit_surface.npy", result.first_hit_surface.detach().cpu().numpy())
        diff = (result.image - reference.image).abs()
        save_image(out_dir / f"{method}_absdiff.png", (diff * 5.0).clamp(0, 1))
        row = {
            "method": method,
            "frame_ms": ms,
            "frame_ms_samples": json.dumps(timing),
            "psnr_reflect": psnr(result.image, reference.image, reference.mirror_mask),
            "psnr_ref_radiance_hit": psnr(result.image, reference.image, ref_radiance),
            "psnr_radiance_union_hit": psnr(result.image, reference.image, ref_radiance | radiance_hit_mask(result.first_hit_surface, roles)),
            "psnr_all_hit_union": psnr(result.image, reference.image, torch.isfinite(result.depth) | torch.isfinite(reference.first_hit_tau)),
            "ssim_reflect": ssim(result.image, reference.image, reference.mirror_mask),
            "lpips": maybe_lpips(result.image, reference.image, lpips_enabled),
            "depth_agreement": depth_agreement(result.depth, reference.first_hit_tau, reference.mirror_mask, tol=cfg["benchmark"].get("depth_tol", 0.02)),
        }
        row.update(visibility_metrics(result.depth, reference.first_hit_tau, result.first_hit_surface, reference.first_hit_surface, reference.mirror_mask, tol=cfg["benchmark"].get("depth_tol", 0.02), surface_roles=roles))
        row.update(result.stats)
        if method.startswith("adaptive_") and bool(cfg["benchmark"].get("require_adaptive_tolerance", False)):
            exhausted = int(result.stats.get("adaptive_tolerance_exhausted_tiles", 0))
            topology_violations = int(result.stats.get("adaptive_final_topology_tolerance_violations", 0))
            if exhausted or topology_violations:
                raise AssertionError(
                    f"{method}: adaptive tolerance not satisfied "
                    f"(exhausted_tiles={exhausted}, final_topology_violations={topology_violations})"
                )
        if (method.endswith("tri") or method.endswith("tri_uv")) and msaa == 4:
            hi = evaluate(method, high_res=True)
            anti = F.avg_pool2d(hi.image.permute(2,0,1)[None],2).squeeze(0).permute(1,2,0)
            anti_ref = F.avg_pool2d(msaa_ref.image.permute(2,0,1)[None],2).squeeze(0).permute(1,2,0)
            anti_mask = F.avg_pool2d(msaa_ref.mirror_mask.float()[None,None],2).squeeze() > 0
            save_image(out_dir / f"{method}_msaa4.png", anti)
            row["psnr_reflect_msaa4"] = psnr(anti, anti_ref, anti_mask)
            row["ssim_reflect_msaa4"] = ssim(anti, anti_ref, anti_mask)
            msaa_images[method] = anti
            msaa_radiance_masks[method] = F.max_pool2d(radiance_hit_mask(hi.first_hit_surface, roles).float()[None,None],2).squeeze() > 0
        if parity_gate:
            other_backend = "torch" if backend in ("cupy", "auto") else "cupy"
            other = evaluate(method, chosen_backend=other_backend)
            mask_diff = int((torch.isfinite(result.depth) != torch.isfinite(other.depth)).sum().item())
            row["backend_mask_mismatches"] = mask_diff
            if mask_diff:
                raise AssertionError(f"{method}: CuPy/Torch hit masks differ at {mask_diff} pixels")
            id_diff = int((result.first_hit_surface != other.first_hit_surface).sum().item())
            row["backend_surface_id_mismatches"] = id_diff
            if id_diff:
                raise AssertionError(f"{method}: CuPy/Torch first-hit surface IDs differ at {id_diff} pixels")
            color_max_error = float((result.image - other.image).abs().max().item())
            row["backend_color_max_error"] = color_max_error
            if color_max_error > 0.01:
                raise AssertionError(f"{method}: CuPy/Torch colors differ by {color_max_error:.4g}")
            if method.endswith("ewa") and (result.stats["pixel_touches"] == 0 or other.stats["pixel_touches"] == 0):
                raise AssertionError(f"{method}: EWA has zero pixel touches")
            for name, candidate in ((backend, result), (other_backend, other)):
                hit = torch.isfinite(candidate.depth)
                if not bool(hit.any()) or not bool((candidate.image[hit].amax() > 0.05)):
                    raise AssertionError(f"{method}: {name} has no colored reflected source pixels")
        rows.append(row)
        results[method] = result
        quality_rows[method] = row
    if "exact_tri_uv" in results:
        exact_uv = results["exact_tri_uv"]
        for candidate_name in ("adaptive_pred_tri_uv", "adaptive_newton_tri_uv"):
            if candidate_name not in results:
                continue
            candidate = results[candidate_name]
            common = radiance_hit_mask(exact_uv.first_hit_surface, roles) & radiance_hit_mask(candidate.first_hit_surface, roles)
            union = radiance_hit_mask(exact_uv.first_hit_surface, roles) | radiance_hit_mask(candidate.first_hit_surface, roles)
            row = quality_rows[candidate_name]
            prefix = "direct_adaptive_vs_exact_uv"
            row[f"{prefix}_psnr_common_hit"] = psnr(candidate.image, exact_uv.image, common)
            row[f"{prefix}_psnr_union_hit"] = psnr(candidate.image, exact_uv.image, union)
            row[f"{prefix}_max_abs_union_hit"] = float((candidate.image - exact_uv.image).abs()[union].max()) if bool(union.any()) else float("nan")
            row[f"{prefix}_radiance_mask_disagreement"] = int((radiance_hit_mask(exact_uv.first_hit_surface, roles) != radiance_hit_mask(candidate.first_hit_surface, roles)).sum().item())
            direct_vis = visibility_metrics(
                candidate.depth,
                exact_uv.depth,
                candidate.first_hit_surface,
                exact_uv.first_hit_surface,
                reference.mirror_mask,
                tol=cfg["benchmark"].get("depth_tol", 0.02),
                surface_roles=roles,
            )
            for key, value in direct_vis.items():
                row[f"{prefix}_{key}"] = value

            def reduction_ratio(stat: str) -> float:
                ref_value = float(exact_uv.stats.get(stat, 0))
                candidate_value = float(candidate.stats.get(stat, 0))
                if candidate_value <= 0.0:
                    return float("inf") if ref_value > 0.0 else float("nan")
                return ref_value / candidate_value

            for stat in (
                "exact_anchor_solves",
                "generated_vertices",
                "generated_triangles",
                "indexed_mesh_payload_bytes",
                "dynamic_raster_payload_bytes",
                "pixel_touches",
            ):
                row[f"{prefix}_{stat}_reduction_x"] = reduction_ratio(stat)
            if "exact_tri_uv" in msaa_images and candidate_name in msaa_images:
                union_msaa = msaa_radiance_masks["exact_tri_uv"] | msaa_radiance_masks[candidate_name]
                row[f"{prefix}_psnr_msaa4_union_hit"] = psnr(msaa_images[candidate_name], msaa_images["exact_tri_uv"], union_msaa)
            save_image(out_dir / f"{candidate_name}_vs_exact_tri_uv_absdiff.png", ((candidate.image - exact_uv.image).abs()*5).clamp(0,1))

    if "exact_tri" in results and "tiled_tri" in results:
        exact = results["exact_tri"]
        tiled = results["tiled_tri"]
        common = radiance_hit_mask(exact.first_hit_surface, roles) & radiance_hit_mask(tiled.first_hit_surface, roles)
        union = radiance_hit_mask(exact.first_hit_surface, roles) | radiance_hit_mask(tiled.first_hit_surface, roles)
        row = quality_rows["tiled_tri"]
        row["direct_tiled_vs_exact_tri_psnr_common_hit"] = psnr(tiled.image, exact.image, common)
        row["direct_tiled_vs_exact_tri_psnr_union_hit"] = psnr(tiled.image, exact.image, union)
        row["direct_tiled_vs_exact_tri_max_abs_union_hit"] = float((tiled.image - exact.image).abs()[union].max()) if bool(union.any()) else float("nan")
        row["direct_tiled_vs_exact_tri_radiance_mask_disagreement"] = int((radiance_hit_mask(exact.first_hit_surface, roles) != radiance_hit_mask(tiled.first_hit_surface, roles)).sum().item())
        if "exact_tri" in msaa_images and "tiled_tri" in msaa_images:
            union_msaa = msaa_radiance_masks["exact_tri"] | msaa_radiance_masks["tiled_tri"]
            row["direct_tiled_vs_exact_tri_psnr_msaa4_union_hit"] = psnr(msaa_images["tiled_tri"], msaa_images["exact_tri"], union_msaa)
            mirror_msaa = F.max_pool2d(msaa_ref.mirror_mask.float()[None,None],2).squeeze() > 0
            row["direct_tiled_vs_exact_tri_psnr_msaa4_reflect"] = psnr(msaa_images["tiled_tri"], msaa_images["exact_tri"], mirror_msaa)
        save_image(out_dir / "tiled_vs_exact_tri_absdiff.png", ((tiled.image - exact.image).abs()*5).clamp(0,1))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--raster-backend", choices=("auto", "torch", "cupy"), default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.raster_backend is not None:
        cfg["benchmark"]["raster_backend"] = args.raster_backend
    device = _device(args.device)
    seed_all(int(cfg.get("seed", 1234)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output or cfg.get("output_dir", f"runs/{stamp}"))
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "system.json", _system_info())
    Path(root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    all_rows = []
    variants = expand_sweep(cfg)
    for i, variant in enumerate(variants):
        vdir = root / f"case_{i:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        Path(vdir / "config.yaml").write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        rows = _run_one(variant, vdir, device)
        tag = variant.get("_sweep_tag", "")
        for r in rows:
            r["case"] = i
            r["sweep_tag"] = tag
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(vdir / "metrics.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))

    df = pd.DataFrame(all_rows)
    df.to_csv(root / "metrics_all.csv", index=False)
    print(f"\nWrote {root / 'metrics_all.csv'}")


if __name__ == "__main__":
    main()
