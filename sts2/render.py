from __future__ import annotations

from dataclasses import dataclass
import torch

from raster import rasterize_ewa, rasterize_triangles, rasterize_uv_triangles
from .camera import PinholeCamera
from .geometry import HeightFieldReflector, PlaneSurface, make_source_grid, grid_triangles
from .tiles import transport_uniform_tiles
from .adaptive import AdaptiveTransportConfig, build_adaptive_transport_mesh
from .transport import solve_specular, differentials, specular_terms

Tensor = torch.Tensor


@dataclass
class RenderResult:
    image: Tensor
    depth: Tensor
    first_hit_surface: Tensor
    stats: dict


def _valid_at_u(reflector, surface, camera, s, u, base_valid):
    c, _, _, _, _ = camera.matrices(device=s.device, dtype=s.dtype)
    q = specular_terms(reflector, surface, u, s, c)
    n = reflector.normal(u)
    same = ((n * q["ell"]).sum(-1) > 0) & ((n * q["d"]).sum(-1) > 0)
    return base_valid & reflector.in_bounds(u) & same


def _tiled_local_differentials(reflector, surface, camera, s, tile_result):
    idx = tile_result.tile_index
    j_us = tile_result.j_us[idx]
    c, _, _, _, _ = camera.matrices(device=s.device, dtype=s.dtype)
    q = specular_terms(reflector, surface, tile_result.u, s, c)
    jpi = camera.jacobian(q["x"])
    jphi = torch.einsum("...ij,...jk,...kl->...il", jpi, q["jx"], j_us)
    tau = q["r_y"]
    j_tau_s = torch.einsum("...i,...ij->...j", q["ell"], q["jy"] - torch.einsum("...ij,...jk->...ik", q["jx"], j_us))
    depth_grad = torch.einsum("...i,...ij->...j", j_tau_s, torch.linalg.pinv(jphi, rcond=1e-8))
    return q, jphi, tau, depth_grad


def render_ewa(
    reflector: HeightFieldReflector,
    camera: PinholeCamera,
    surfaces: list[PlaneSurface],
    mirror_mask: Tensor,
    *,
    n_samples: int,
    mode: str,
    tiles_per_axis: int = 8,
    backend: str = "auto",
    environment_rgb=(0.03, 0.04, 0.06),
    kernel_scale: float = 0.65,
    max_radius: int = 8,
) -> RenderResult:
    device = mirror_mask.device
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    all_p=[]; all_tau=[]; all_color=[]; all_cov=[]; all_grad=[]; all_rad=[]; all_ids=[]
    stats={"exact_anchor_solves":0,"source_samples":0,"max_tile_error_est":0.0,"invalid_samples":0,"transport_anchor_payload_bytes":0,"dynamic_raster_payload_bytes":0}
    eye2=torch.eye(2,device=device,dtype=dtype)
    c, _, _, _, _ = camera.matrices(device=device, dtype=dtype)
    for surface_id, surf in enumerate(surfaces):
        s=make_source_grid(n_samples,device=device,dtype=dtype,vertices=False)
        if mode=="exact":
            sol=solve_specular(reflector,surf,s,c)
            diff=differentials(reflector,surf,camera,s,sol.u)
            u=sol.u; valid=sol.valid; jphi=diff.j_phi; tau=diff.tau; grad=diff.depth_grad
            stats["exact_anchor_solves"] += s.shape[0]
            stats["transport_anchor_payload_bytes"] += int(s.shape[0] * (2 + 4) * 4)
        elif mode=="tiled":
            tr=transport_uniform_tiles(reflector,surf,camera,s,tiles_per_axis)
            q,jphi,tau,grad=_tiled_local_differentials(reflector,surf,camera,s,tr)
            u=tr.u; valid=_valid_at_u(reflector,surf,camera,s,u,tr.valid)
            stats["exact_anchor_solves"] += tr.exact_anchor_solves
            stats["transport_anchor_payload_bytes"] += int(tr.anchors_s.shape[0] * (2 + 4) * 4)
            stats["max_tile_error_est"] = max(stats["max_tile_error_est"], float(tr.estimated_pixel_error.max().detach().cpu()))
        else:
            raise ValueError(mode)
        x,_,_=reflector.eval(u)
        p,z=camera.project(x)
        valid &= z>0
        y,_=surf.eval(s)
        color=surf.radiance(s)
        spacing=2.0/n_samples
        cs=((kernel_scale*spacing)**2)*eye2.expand(s.shape[0],2,2)
        cov=torch.einsum("...ij,...jk,...lk->...il",jphi,cs,jphi) + (0.35**2)*eye2
        # Keep invalid splats off-screen with radiance disabled; easier concatenation.
        p=torch.where(valid[:,None],p,torch.full_like(p,-1e6))
        tau=torch.where(valid,tau,torch.full_like(tau,1e6))
        rad=torch.full((s.shape[0],),surf.role=="radiance",device=device,dtype=torch.bool) & valid
        all_p.append(p.float()); all_tau.append(tau.float()); all_color.append(color.float()); all_cov.append(cov.float()); all_grad.append(grad.float()); all_rad.append(rad)
        all_ids.append(torch.full((s.shape[0],),surface_id,device=device,dtype=torch.int32))
        stats["source_samples"] += s.shape[0]
        stats["invalid_samples"] += int((~valid).sum().item())
    p=torch.cat(all_p); tau=torch.cat(all_tau); color=torch.cat(all_color); cov=torch.cat(all_cov); grad=torch.cat(all_grad); rad=torch.cat(all_rad)
    ids=torch.cat(all_ids)
    stats["dynamic_raster_payload_bytes"] = int(sum(t.numel() * t.element_size() for t in (p,tau,color,cov,grad,rad)))
    img,depth,first_id,rstats=rasterize_ewa(p,tau,color,cov,grad,rad,camera.height,camera.width,mirror_mask=mirror_mask,environment_rgb=environment_rgb,max_radius=max_radius,surface_ids=ids,backend=backend)
    stats.update(rstats)
    return RenderResult(img,depth,first_id,stats)


def render_triangles(
    reflector: HeightFieldReflector,
    camera: PinholeCamera,
    surfaces: list[PlaneSurface],
    mirror_mask: Tensor,
    *,
    n_cells: int,
    mode: str,
    tiles_per_axis: int = 8,
    backend: str = "auto",
    environment_rgb=(0.03,0.04,0.06),
) -> RenderResult:
    device=mirror_mask.device
    dtype=torch.float32 if device.type=="cuda" else torch.float64
    all_tp=[]; all_tt=[]; all_tc=[]; all_tr=[]; all_ids=[]
    stats={"exact_anchor_solves":0,"source_vertices":0,"triangles":0,"max_tile_error_est":0.0,"invalid_triangles":0,"transport_anchor_payload_bytes":0,"dynamic_raster_payload_bytes":0}
    c,_,_,_,_=camera.matrices(device=device,dtype=dtype)
    for surface_id, surf in enumerate(surfaces):
        s=make_source_grid(n_cells,device=device,dtype=dtype,vertices=True)
        if mode=="exact":
            sol=solve_specular(reflector,surf,s,c)
            u=sol.u; valid=sol.valid
            stats["exact_anchor_solves"] += s.shape[0]
            stats["transport_anchor_payload_bytes"] += int(s.shape[0] * (2 + 4) * 4)
        elif mode=="tiled":
            tr=transport_uniform_tiles(reflector,surf,camera,s,tiles_per_axis)
            u=tr.u; valid=_valid_at_u(reflector,surf,camera,s,u,tr.valid)
            stats["exact_anchor_solves"] += tr.exact_anchor_solves
            stats["transport_anchor_payload_bytes"] += int(tr.anchors_s.shape[0] * (2 + 4) * 4)
            stats["max_tile_error_est"] = max(stats["max_tile_error_est"], float(tr.estimated_pixel_error.max().detach().cpu()))
        else:
            raise ValueError(mode)
        x,_,_=reflector.eval(u)
        p,z=camera.project(x)
        y,_=surf.eval(s)
        tau=torch.linalg.vector_norm(y-x,dim=-1)
        color=surf.radiance(s)
        valid &= z>0
        tri=grid_triangles(n_cells,device=device)
        tv=valid[tri].all(dim=-1)
        tp=p[tri][tv].float(); tt=tau[tri][tv].float(); tc=color[tri][tv].float()
        tr=torch.full((tp.shape[0],),surf.role=="radiance",device=device,dtype=torch.bool)
        all_tp.append(tp); all_tt.append(tt); all_tc.append(tc); all_tr.append(tr)
        all_ids.append(torch.full((tp.shape[0],),surface_id,device=device,dtype=torch.int32))
        stats["source_vertices"] += s.shape[0]
        stats["triangles"] += tp.shape[0]
        stats["invalid_triangles"] += int((~tv).sum().item())
    if not all_tp:
        raise RuntimeError("no triangles")
    tp=torch.cat(all_tp); tt=torch.cat(all_tt); tc=torch.cat(all_tc); tr=torch.cat(all_tr)
    ids=torch.cat(all_ids)
    stats["dynamic_raster_payload_bytes"] = int(sum(t.numel() * t.element_size() for t in (tp,tt,tc,tr)))
    img,depth,first_id,rstats=rasterize_triangles(tp,tt,tc,tr,camera.height,camera.width,mirror_mask=mirror_mask,environment_rgb=environment_rgb,surface_ids=ids,backend=backend)
    stats.update(rstats)
    return RenderResult(img,depth,first_id,stats)


def _merge_adaptive_stats(total: dict, part: dict) -> None:
    """Aggregate per-surface adaptive diagnostics conservatively."""
    min_keys = {"min_sigma_min_jphi", "min_abs_det_jphi"}
    max_keys = {"max_kappa_fu", "adaptive_max_depth_reached"}
    for key, value in part.items():
        if not isinstance(value, (int, float)):
            continue
        if key in min_keys:
            total[key] = min(total.get(key, float("inf")), value)
        elif key in max_keys or key.endswith("_max") or "_p99" in key or "_p95" in key or "_p50" in key:
            total[key] = max(total.get(key, float("-inf")), value)
        else:
            total[key] = total.get(key, 0) + value


def _shade_uv_first_hits(
    surfaces: list[PlaneSurface],
    first_id: Tensor,
    uv: Tensor,
    uv_grad: Tensor,
    mirror_mask: Tensor,
    *,
    environment_rgb: tuple[float, float, float],
    filter_taps: int,
) -> Tensor:
    dtype = uv.dtype
    device = uv.device
    env = torch.tensor(environment_rgb, device=device, dtype=dtype)
    image = env.expand(*first_id.shape, 3).clone()
    has_hit = first_id >= 0
    image[has_hit] = 0.0
    for surface_id, surface in enumerate(surfaces):
        if surface.role != "radiance":
            continue
        mask = first_id == surface_id
        if not bool(mask.any()):
            continue
        image[mask] = surface.radiance_filtered(uv[mask], uv_grad[mask], taps=filter_taps)
    image = torch.where(mirror_mask[..., None], image, torch.zeros_like(image))
    return image


def render_uv_triangles(
    reflector: HeightFieldReflector,
    camera: PinholeCamera,
    surfaces: list[PlaneSurface],
    mirror_mask: Tensor,
    *,
    mode: str,
    n_cells: int = 64,
    adaptive_cfg: AdaptiveTransportConfig | None = None,
    backend: str = "auto",
    environment_rgb=(0.03, 0.04, 0.06),
    filter_taps: int = 4,
) -> RenderResult:
    """Render exact or adaptive reflected triangles carrying source coordinates.

    Unlike render_triangles(), this path never bakes source RGB into vertices.
    Visibility is resolved first in reflected depth, then the shared source field
    is sampled per winning fragment using the affine source-coordinate footprint.
    """
    device = mirror_mask.device
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    c, _, _, _, _ = camera.matrices(device=device, dtype=dtype)
    all_tp: list[Tensor] = []
    all_tt: list[Tensor] = []
    all_ts: list[Tensor] = []
    all_tr: list[Tensor] = []
    all_ids: list[Tensor] = []
    stats: dict[str, float | int] = {
        "exact_anchor_solves": 0,
        "validation_exact_solves": 0,
        "source_vertices": 0,
        "generated_vertices": 0,
        "triangles": 0,
        "generated_triangles": 0,
        "invalid_triangles": 0,
        "transport_anchor_payload_bytes": 0,
        "dynamic_raster_payload_bytes": 0,
        "indexed_mesh_payload_bytes": 0,
        "persistent_transport_bytes": 0,
        "source_filter_taps": filter_taps,
    }

    if mode == "adaptive" and adaptive_cfg is None:
        adaptive_cfg = AdaptiveTransportConfig()

    for surface_id, surf in enumerate(surfaces):
        if mode == "exact":
            s = make_source_grid(n_cells, device=device, dtype=dtype, vertices=True)
            sol = solve_specular(reflector, surf, s, c)
            u = sol.u
            x, _, _ = reflector.eval(u)
            p, z = camera.project(x)
            y, _ = surf.eval(s)
            tau = torch.linalg.vector_norm(y - x, dim=-1)
            tri = grid_triangles(n_cells, device=device)
            valid = sol.valid & (z > 0)
            tv = valid[tri].all(dim=-1)
            tp = p[tri][tv]
            tt = tau[tri][tv]
            ts = s[tri][tv]
            stats["exact_anchor_solves"] += int(s.shape[0])
            stats["source_vertices"] += int(s.shape[0])
            stats["generated_vertices"] += int(s.shape[0])
            stats["triangles"] += int(tv.sum().item())
            stats["generated_triangles"] += int(tv.sum().item())
            stats["invalid_triangles"] += int((~tv).sum().item())
            stats["transport_anchor_payload_bytes"] += int(s.shape[0] * (2 + 4) * 4)
            stats["indexed_mesh_payload_bytes"] += int(
                s.numel() * s.element_size()
                + p.numel() * p.element_size()
                + tau.numel() * tau.element_size()
                + tri.numel() * tri.element_size()
            )
        elif mode == "adaptive":
            assert adaptive_cfg is not None
            mesh = build_adaptive_transport_mesh(
                reflector, surf, camera, adaptive_cfg, device=device, dtype=dtype
            )
            tri = mesh.triangles[mesh.triangle_valid]
            tp = mesh.screen_vertices[tri]
            tt = mesh.tau_vertices[tri]
            ts = mesh.source_vertices[tri]
            stats["source_vertices"] += int(mesh.source_vertices.shape[0])
            stats["triangles"] += int(tri.shape[0])
            stats["transport_anchor_payload_bytes"] += int(mesh.leaf_count * (2 + 2 + 4) * 4)
            stats["indexed_mesh_payload_bytes"] += int(
                mesh.source_vertices.numel() * mesh.source_vertices.element_size()
                + mesh.screen_vertices.numel() * mesh.screen_vertices.element_size()
                + mesh.tau_vertices.numel() * mesh.tau_vertices.element_size()
                + mesh.triangles.numel() * mesh.triangles.element_size()
            )
            _merge_adaptive_stats(stats, mesh.stats)
        else:
            raise ValueError(f"unknown UV triangle mode: {mode}")

        if tp.numel() == 0:
            continue
        rad = torch.full((tp.shape[0],), surf.role == "radiance", device=device, dtype=torch.bool)
        ids = torch.full((tp.shape[0],), surface_id, device=device, dtype=torch.int32)
        all_tp.append(tp.float())
        all_tt.append(tt.float())
        all_ts.append(ts.float())
        all_tr.append(rad)
        all_ids.append(ids)

    if not all_tp:
        env = torch.tensor(environment_rgb, device=device, dtype=torch.float32)
        image = torch.where(mirror_mask[..., None], env.expand(*mirror_mask.shape, 3), torch.zeros((*mirror_mask.shape, 3), device=device, dtype=torch.float32))
        return RenderResult(
            image=image,
            depth=torch.full(mirror_mask.shape, float("inf"), device=device, dtype=torch.float32),
            first_hit_surface=torch.full(mirror_mask.shape, -1, device=device, dtype=torch.int32),
            stats=stats,
        )

    tp = torch.cat(all_tp)
    tt = torch.cat(all_tt)
    ts = torch.cat(all_ts)
    tr = torch.cat(all_tr)
    ids = torch.cat(all_ids)
    stats["dynamic_raster_payload_bytes"] = int(
        sum(t.numel() * t.element_size() for t in (tp, tt, ts, tr, ids))
    )
    depth, first_id, uv, uv_grad, rstats = rasterize_uv_triangles(
        tp, tt, ts, tr,
        camera.height, camera.width,
        mirror_mask=mirror_mask,
        surface_ids=ids,
        backend=backend,
    )
    stats.update(rstats)
    image = _shade_uv_first_hits(
        surfaces, first_id, uv, uv_grad, mirror_mask,
        environment_rgb=environment_rgb,
        filter_taps=filter_taps,
    )
    return RenderResult(image, depth, first_id, stats)
