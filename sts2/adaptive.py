from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .camera import PinholeCamera
from .geometry import HeightFieldReflector, PlaneSurface
from .transport import differentials, newton_correct, solve_specular, specular_terms

Tensor = torch.Tensor


@dataclass(frozen=True)
class AdaptiveTransportConfig:
    """Controls the Stage-A1 adaptive source-chart tessellation.

    max_depth is relative to each base tile. The source chart is [-1, 1]^2.
    The split tolerances are diagnostics on the *rendered map*, not only on the
    stationary-root residual.
    """

    base_tiles_per_axis: int = 4
    max_depth: int = 4
    screen_error_px: float = 0.10
    depth_error: float = 2.0e-3
    root_error_px: float = 0.025
    source_uv_error: float = 2.0e-3
    radiance_error: float = 0.02
    max_kappa_fu: float = 1.0e5
    min_sigma_jphi: float = 1.0e-4
    min_abs_det_jphi: float = 1.0e-6
    newton_corrections: int = 1
    max_leaf_tiles: int = 20000
    validate_exact_probes: bool = False
    validation_stride: int = 8

    def __post_init__(self) -> None:
        if self.base_tiles_per_axis < 1:
            raise ValueError("base_tiles_per_axis must be >= 1")
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.newton_corrections not in (0, 1):
            raise ValueError("newton_corrections must be 0 or 1")
        if self.validation_stride < 1:
            raise ValueError("validation_stride must be >= 1")


@dataclass(frozen=True)
class _Cell:
    x0: int
    y0: int
    size: int
    depth: int
    init_u: Tensor | None = None


@dataclass
class _Leaf:
    cell: _Cell
    anchor_s: Tensor
    anchor_u: Tensor
    j_us: Tensor
    anchor_valid: bool
    max_root_error_px: float
    max_screen_interp_error_px: float
    max_depth_interp_error: float
    max_source_uv_error: float
    max_radiance_error: float
    screen_probe_inside: bool
    max_kappa_fu: float
    min_sigma_jphi: float
    min_abs_det_jphi: float
    orientation_consistent: bool
    probe_valid: bool
    tolerance_met: bool


@dataclass
class AdaptiveMeshResult:
    source_vertices: Tensor
    reflector_u: Tensor
    screen_vertices: Tensor
    tau_vertices: Tensor
    triangles: Tensor
    triangle_valid: Tensor
    leaf_count: int
    stats: dict[str, float | int]


def _source_from_grid(x: int, y: int, scale: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.tensor(
        [-1.0 + 2.0 * float(x) / float(scale), -1.0 + 2.0 * float(y) / float(scale)],
        device=device,
        dtype=dtype,
    )


def _cell_anchor(cell: _Cell, scale: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    half = cell.size // 2
    return _source_from_grid(cell.x0 + half, cell.y0 + half, scale, device=device, dtype=dtype)


def _cell_corners(cell: _Cell, scale: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    pts = (
        (cell.x0, cell.y0),
        (cell.x0 + cell.size, cell.y0),
        (cell.x0 + cell.size, cell.y0 + cell.size),
        (cell.x0, cell.y0 + cell.size),
    )
    return torch.stack([_source_from_grid(x, y, scale, device=device, dtype=dtype) for x, y in pts], dim=0)


def _cell_probes(cell: _Cell, scale: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return center plus four edge midpoints and their local [0,1]^2 coordinates."""
    h = cell.size // 2
    pts = (
        (cell.x0 + h, cell.y0 + h),
        (cell.x0 + h, cell.y0),
        (cell.x0 + cell.size, cell.y0 + h),
        (cell.x0 + h, cell.y0 + cell.size),
        (cell.x0, cell.y0 + h),
    )
    local = torch.tensor(
        [[0.5, 0.5], [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],
        device=device,
        dtype=dtype,
    )
    s = torch.stack([_source_from_grid(x, y, scale, device=device, dtype=dtype) for x, y in pts], dim=0)
    return s, local


def _predict(anchor_s: Tensor, anchor_u: Tensor, j_us: Tensor, s: Tensor) -> Tensor:
    ds = s - anchor_s
    return anchor_u + torch.einsum("ij,nj->ni", j_us, ds)


def _physical_valid(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera_center: Tensor,
    s: Tensor,
    u: Tensor,
) -> Tensor:
    q = specular_terms(reflector, surface, u, s, camera_center)
    n = reflector.normal(u)
    same_side = ((n * q["ell"]).sum(dim=-1) > 0) & ((n * q["d"]).sum(dim=-1) > 0)
    finite = torch.isfinite(u).all(dim=-1) & torch.isfinite(q["F"]).all(dim=-1)
    return finite & reflector.in_bounds(u) & same_side


def _project_and_tau(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    s: Tensor,
    u: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    x, _, _ = reflector.eval(u)
    p, z = camera.project(x)
    y, _ = surface.eval(s)
    tau = torch.linalg.vector_norm(y - x, dim=-1)
    return p, z, tau


def _triangle_interpolate_corners(values: Tensor, local: Tensor) -> Tensor:
    """Interpolate four corner values with the same BL-BR-TR / BL-TR-TL diagonal used by the mesh."""
    bl, br, tr, tl = values.unbind(dim=0)
    x = local[:, 0]
    y = local[:, 1]
    lower = x >= y
    tail_shape = (1,) * (values.ndim - 1)
    xw = x.reshape(-1, *tail_shape)
    yw = y.reshape(-1, *tail_shape)
    low = (1.0 - xw) * bl + (xw - yw) * br + yw * tr
    high = (1.0 - yw) * bl + xw * tr + (yw - xw) * tl
    mask = lower.reshape(-1, *tail_shape)
    return torch.where(mask, low, high)


def _select_probe_triangles(values: Tensor, local: Tensor) -> Tensor:
    """Select the BL-BR-TR or BL-TR-TL corner triangle for each source probe."""
    bl, br, tr, tl = values.unbind(dim=0)
    lower = local[:, 0] >= local[:, 1]
    low_tri = torch.stack((bl, br, tr), dim=0)
    high_tri = torch.stack((bl, tr, tl), dim=0)
    # Expand to one triangle per probe.
    low_all = low_tri.unsqueeze(0).expand(local.shape[0], *low_tri.shape)
    high_all = high_tri.unsqueeze(0).expand(local.shape[0], *high_tri.shape)
    mask = lower.reshape(-1, *([1] * (low_all.ndim - 1)))
    return torch.where(mask, low_all, high_all)


def _screen_barycentric(tri_p: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
    a = tri_p[:, 0]
    b = tri_p[:, 1]
    c = tri_p[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    safe = torch.where(den.abs() < 1e-10, torch.ones_like(den), den)
    w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe
    w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe
    w0 = 1.0 - w1 - w2
    weights = torch.stack((w0, w1, w2), dim=-1)
    inside = (den.abs() >= 1e-10) & (weights >= -1e-4).all(dim=-1) & (weights <= 1.0001).all(dim=-1)
    return weights, inside


def _weighted_triangle(values: Tensor, weights: Tensor) -> Tensor:
    tail = (1,) * (values.ndim - 2)
    return (values * weights.reshape(weights.shape[0], 3, *tail)).sum(dim=1)


def _det2(a: Tensor) -> Tensor:
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]


def _quantiles(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_p50": float("nan"), f"{prefix}_p95": float("nan"), f"{prefix}_p99": float("nan"), f"{prefix}_max": float("nan")}
    t = torch.tensor(values, dtype=torch.float64)
    return {
        f"{prefix}_p50": float(torch.quantile(t, 0.50)),
        f"{prefix}_p95": float(torch.quantile(t, 0.95)),
        f"{prefix}_p99": float(torch.quantile(t, 0.99)),
        f"{prefix}_max": float(t.max()),
    }


def _evaluate_cell(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    camera_center: Tensor,
    cell: _Cell,
    scale: int,
    cfg: AdaptiveTransportConfig,
) -> _Leaf:
    device, dtype = camera_center.device, camera_center.dtype
    anchor_s = _cell_anchor(cell, scale, device=device, dtype=dtype).unsqueeze(0)
    init = cell.init_u.unsqueeze(0) if cell.init_u is not None else None
    sol = solve_specular(reflector, surface, anchor_s, camera_center, init_u=init)
    anchor_u = sol.u[0]
    anchor_valid = bool(sol.valid[0].item())

    try:
        anchor_diff = differentials(reflector, surface, camera, anchor_s, sol.u)
        j_us = anchor_diff.j_us[0]
        anchor_det_t = _det2(anchor_diff.j_phi)[0]
        finite_anchor_diff = (
            torch.isfinite(j_us).all()
            and torch.isfinite(anchor_det_t)
            and torch.isfinite(anchor_diff.kappa_fu[0])
            and torch.isfinite(anchor_diff.sigma_min_jphi[0])
        )
        anchor_det = float(anchor_det_t.detach().cpu())
        anchor_kappa = float(anchor_diff.kappa_fu[0].detach().cpu())
        anchor_sigma = float(anchor_diff.sigma_min_jphi[0].detach().cpu())
        anchor_valid = anchor_valid and bool(finite_anchor_diff.item())
    except RuntimeError:
        # Preserve a finite payload so the cell can be rejected or subdivided.
        j_us = torch.zeros((2, 2), device=device, dtype=dtype)
        anchor_det = 0.0
        anchor_kappa = float("inf")
        anchor_sigma = 0.0
        anchor_valid = False

    corners_s = _cell_corners(cell, scale, device=device, dtype=dtype)
    probes_s, probes_local = _cell_probes(cell, scale, device=device, dtype=dtype)
    all_s = torch.cat((corners_s, probes_s), dim=0)
    pred_u = _predict(anchor_s[0], anchor_u, j_us, all_s)
    production_u = pred_u
    if cfg.newton_corrections:
        production_u, _ = newton_correct(reflector, surface, all_s, camera_center, production_u)

    # The next Newton displacement estimates the root error remaining after the
    # production path. It is never applied to production vertices/probes.
    _, du_next = newton_correct(reflector, surface, all_s, camera_center, production_u)
    q_prod = specular_terms(reflector, surface, production_u, all_s, camera_center)
    jpi_prod = camera.jacobian(q_prod["x"])
    dp_root = torch.einsum("...ij,...jk,...k->...i", jpi_prod, q_prod["jx"], du_next)
    root_error = torch.linalg.vector_norm(dp_root, dim=-1)

    corners_u = production_u[:4]
    probes_u = production_u[4:]
    corners_p, corners_z, corners_tau = _project_and_tau(reflector, surface, camera, corners_s, corners_u)
    probes_p, probes_z, probes_tau = _project_and_tau(reflector, surface, camera, probes_s, probes_u)
    interp_p = _triangle_interpolate_corners(corners_p, probes_local)
    screen_interp_error = torch.linalg.vector_norm(probes_p - interp_p, dim=-1)

    # What the rasterizer actually reconstructs at the physical probe pixel is
    # governed by *screen* barycentrics, not by the source-space probe weights.
    probe_tri_p = _select_probe_triangles(corners_p, probes_local)
    probe_tri_tau = _select_probe_triangles(corners_tau, probes_local)
    probe_tri_s = _select_probe_triangles(corners_s, probes_local)
    screen_weights, screen_inside = _screen_barycentric(probe_tri_p, probes_p)
    raster_tau = _weighted_triangle(probe_tri_tau, screen_weights)
    raster_s = _weighted_triangle(probe_tri_s, screen_weights)
    depth_interp_error = (probes_tau - raster_tau).abs()
    source_uv_error = torch.linalg.vector_norm(probes_s - raster_s, dim=-1)
    radiance_error = (surface.radiance(probes_s) - surface.radiance(raster_s)).abs().amax(dim=-1)

    prod_valid = _physical_valid(reflector, surface, camera_center, all_s, production_u)
    front = torch.cat((corners_z, probes_z), dim=0) > 0
    probe_valid = bool((prod_valid & front).all().item())

    try:
        probe_diff = differentials(reflector, surface, camera, all_s, production_u)
        dets = _det2(probe_diff.j_phi)
        finite_probe_diff = (
            torch.isfinite(dets).all()
            and torch.isfinite(probe_diff.kappa_fu).all()
            and torch.isfinite(probe_diff.sigma_min_jphi).all()
        )
        max_kappa = max(anchor_kappa, float(probe_diff.kappa_fu.max().detach().cpu()))
        min_sigma = min(anchor_sigma, float(probe_diff.sigma_min_jphi.min().detach().cpu()))
        min_abs_det = min(abs(anchor_det), float(dets.abs().min().detach().cpu()))
        if abs(anchor_det) <= cfg.min_abs_det_jphi or not bool(finite_probe_diff.item()):
            orientation_consistent = False
        else:
            anchor_sign = 1.0 if anchor_det > 0.0 else -1.0
            orientation_consistent = bool(((dets * anchor_sign) > cfg.min_abs_det_jphi).all().item())
    except RuntimeError:
        max_kappa = float("inf")
        min_sigma = 0.0
        min_abs_det = 0.0
        orientation_consistent = False

    max_root = float(root_error.max().detach().cpu())
    max_screen = float(screen_interp_error.max().detach().cpu())
    max_depth = float(depth_interp_error.max().detach().cpu())
    max_source_uv = float(source_uv_error.max().detach().cpu())
    max_radiance = float(radiance_error.max().detach().cpu())
    screen_probe_inside = bool(screen_inside.all().item())
    appearance_met = (
        surface.role != "radiance"
        or (max_source_uv <= cfg.source_uv_error and max_radiance <= cfg.radiance_error)
    )
    tolerance_met = (
        anchor_valid
        and probe_valid
        and orientation_consistent
        and max_root <= cfg.root_error_px
        and max_screen <= cfg.screen_error_px
        and max_depth <= cfg.depth_error
        and appearance_met
        # An exact edge midpoint can lie just outside its affine edge under
        # smooth curvature. The measured screen/UV/depth errors gate that warp;
        # demanding triangle containment here rejects almost every valid tile.
        and max_kappa <= cfg.max_kappa_fu
        and min_sigma >= cfg.min_sigma_jphi
        and min_abs_det >= cfg.min_abs_det_jphi
    )
    return _Leaf(
        cell=cell,
        anchor_s=anchor_s[0],
        anchor_u=anchor_u,
        j_us=j_us,
        anchor_valid=anchor_valid,
        max_root_error_px=max_root,
        max_screen_interp_error_px=max_screen,
        max_depth_interp_error=max_depth,
        max_source_uv_error=max_source_uv,
        max_radiance_error=max_radiance,
        screen_probe_inside=screen_probe_inside,
        max_kappa_fu=max_kappa,
        min_sigma_jphi=min_sigma,
        min_abs_det_jphi=min_abs_det,
        orientation_consistent=orientation_consistent,
        probe_valid=probe_valid,
        tolerance_met=tolerance_met,
    )


def _child_cells(leaf: _Leaf) -> list[_Cell]:
    c = leaf.cell
    half = c.size // 2
    if half < 1:
        return []
    out = []
    for oy in (0, half):
        for ox in (0, half):
            x0, y0 = c.x0 + ox, c.y0 + oy
            # Each child still receives a fully converged stationary-path anchor.
            out.append(_Cell(x0=x0, y0=y0, size=half, depth=c.depth + 1, init_u=None))
    return out


def _ordered_boundary_keys(cell: _Cell, corner_keys: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = cell.x0, cell.y0, cell.x0 + cell.size, cell.y0 + cell.size
    bottom = sorted((k for k in corner_keys if k[1] == y0 and x0 <= k[0] <= x1), key=lambda q: q[0])
    right = sorted((k for k in corner_keys if k[0] == x1 and y0 < k[1] <= y1), key=lambda q: q[1])
    top = sorted((k for k in corner_keys if k[1] == y1 and x0 <= k[0] < x1), key=lambda q: q[0], reverse=True)
    left = sorted((k for k in corner_keys if k[0] == x0 and y0 < k[1] < y1), key=lambda q: q[1], reverse=True)
    polygon = bottom + right + top + left
    # Deduplicate any accidental repeats while preserving order.
    seen: set[tuple[int, int]] = set()
    return [k for k in polygon if not (k in seen or seen.add(k))]


def _leaf_corner_keys(cell: _Cell) -> tuple[tuple[int, int], ...]:
    return (
        (cell.x0, cell.y0),
        (cell.x0 + cell.size, cell.y0),
        (cell.x0 + cell.size, cell.y0 + cell.size),
        (cell.x0, cell.y0 + cell.size),
    )


def _build_conforming_topology(leaves: list[_Leaf]) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[set[int]], list[int]]:
    corner_keys: set[tuple[int, int]] = set()
    for leaf in leaves:
        corner_keys.update(_leaf_corner_keys(leaf.cell))

    vertex_keys: list[tuple[int, int]] = []
    vertex_index: dict[tuple[int, int], int] = {}
    incident: list[set[int]] = []

    def vid(key: tuple[int, int], leaf_idx: int) -> int:
        if key not in vertex_index:
            vertex_index[key] = len(vertex_keys)
            vertex_keys.append(key)
            incident.append(set())
        i = vertex_index[key]
        incident[i].add(leaf_idx)
        return i

    tris: list[tuple[int, int, int]] = []
    tri_owner: list[int] = []
    for li, leaf in enumerate(leaves):
        cell = leaf.cell
        poly = _ordered_boundary_keys(cell, corner_keys)
        if len(poly) < 4:
            continue
        if len(poly) == 4:
            ids = [vid(k, li) for k in poly]
            # poly is BL, BR, TR, TL for an unsplit boundary.
            tris.append((ids[0], ids[1], ids[2]))
            tri_owner.append(li)
            tris.append((ids[0], ids[2], ids[3]))
            tri_owner.append(li)
            continue
        center = (cell.x0 + cell.size // 2, cell.y0 + cell.size // 2)
        ci = vid(center, li)
        ids = [vid(k, li) for k in poly]
        for a, b in zip(ids, ids[1:] + ids[:1]):
            tris.append((ci, a, b))
            tri_owner.append(li)
    return vertex_keys, tris, incident, tri_owner


def _choose_incident_leaf(leaves: list[_Leaf], choices: Iterable[int]) -> int:
    return min(choices, key=lambda i: (leaves[i].cell.size, i))


def _final_triangle_probe_stats(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    camera_center: Tensor,
    leaves: list[_Leaf],
    source_vertices: Tensor,
    screen_vertices: Tensor,
    tau_vertices: Tensor,
    triangles: Tensor,
    triangle_valid: Tensor,
    tri_owner: list[int],
    cfg: AdaptiveTransportConfig,
) -> dict[str, float | int]:
    """Audit the actual transition triangles, not only each quadtree cell.

    Hanging-node transition cells are triangulated as polygon fans. Their final
    triangles need an independent probe because the cell-level BL-BR-TR /
    BL-TR-TL estimator does not exactly describe those fan wedges. This probe is
    part of the production error accounting and uses no fully converged root.
    """
    if triangles.numel() == 0:
        out: dict[str, float | int] = {
            "adaptive_final_triangle_probe_count": 0,
            "adaptive_final_topology_tolerance_violations": 0,
            "final_topology_production_newton_corrections": 0,
            "final_topology_estimator_newton_corrections": 0,
            "final_topology_newton_linear_solves": 0,
        }
        for prefix in (
            "final_root_screen_error_px",
            "final_triangle_screen_error_px",
            "final_triangle_depth_error",
            "final_source_uv_error",
            "final_radiance_error",
        ):
            out.update(_quantiles([], prefix))
        return out

    tri_s = source_vertices[triangles]
    tri_p = screen_vertices[triangles]
    tri_tau = tau_vertices[triangles]
    probe_s = tri_s.mean(dim=1)
    owners = torch.tensor(tri_owner, device=probe_s.device, dtype=torch.int64)
    anchor_s = torch.stack([leaf.anchor_s for leaf in leaves], dim=0)[owners]
    anchor_u = torch.stack([leaf.anchor_u for leaf in leaves], dim=0)[owners]
    j_us = torch.stack([leaf.j_us for leaf in leaves], dim=0)[owners]
    pred = anchor_u + torch.einsum("nij,nj->ni", j_us, probe_s - anchor_s)
    production = pred
    if cfg.newton_corrections:
        production, _ = newton_correct(reflector, surface, probe_s, camera_center, production)

    _, du_next = newton_correct(reflector, surface, probe_s, camera_center, production)
    q = specular_terms(reflector, surface, production, probe_s, camera_center)
    jpi = camera.jacobian(q["x"])
    dp_root = torch.einsum("...ij,...jk,...k->...i", jpi, q["jx"], du_next)
    root_error = torch.linalg.vector_norm(dp_root, dim=-1)

    probe_p, probe_z, probe_tau = _project_and_tau(reflector, surface, camera, probe_s, production)
    interp_p = tri_p.mean(dim=1)
    screen_error = torch.linalg.vector_norm(probe_p - interp_p, dim=-1)
    weights, screen_inside = _screen_barycentric(tri_p, probe_p)
    raster_tau = _weighted_triangle(tri_tau, weights)
    raster_s = _weighted_triangle(tri_s, weights)
    depth_error = (probe_tau - raster_tau).abs()
    uv_error = torch.linalg.vector_norm(probe_s - raster_s, dim=-1)
    radiance_error = (surface.radiance(probe_s) - surface.radiance(raster_s)).abs().amax(dim=-1)
    physical = _physical_valid(reflector, surface, camera_center, probe_s, production) & (probe_z > 0)
    appearance_ok = torch.ones_like(physical)
    if surface.role == "radiance":
        appearance_ok = (uv_error <= cfg.source_uv_error) & (radiance_error <= cfg.radiance_error)
    within = (
        triangle_valid
        & physical
        & screen_inside
        & (root_error <= cfg.root_error_px)
        & (screen_error <= cfg.screen_error_px)
        & (depth_error <= cfg.depth_error)
        & appearance_ok
    )

    ntri = int(triangles.shape[0])
    out = {
        "adaptive_final_triangle_probe_count": ntri,
        "adaptive_final_topology_tolerance_violations": int((~within).sum().item()),
        "adaptive_final_probe_outside_triangle": int((~screen_inside).sum().item()),
        "adaptive_final_probe_physical_invalid": int((~physical).sum().item()),
        "final_topology_production_newton_corrections": ntri * cfg.newton_corrections,
        "final_topology_estimator_newton_corrections": ntri,
        "final_topology_newton_linear_solves": ntri * (1 + cfg.newton_corrections),
    }
    out.update(_quantiles(root_error.detach().cpu().tolist(), "final_root_screen_error_px"))
    out.update(_quantiles(screen_error.detach().cpu().tolist(), "final_triangle_screen_error_px"))
    out.update(_quantiles(depth_error.detach().cpu().tolist(), "final_triangle_depth_error"))
    out.update(_quantiles(uv_error.detach().cpu().tolist(), "final_source_uv_error"))
    out.update(_quantiles(radiance_error.detach().cpu().tolist(), "final_radiance_error"))
    return out


def _validate_exact_probes(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    camera_center: Tensor,
    leaves: list[_Leaf],
    scale: int,
    cfg: AdaptiveTransportConfig,
) -> dict[str, float | int]:
    root_errors: list[float] = []
    screen_errors: list[float] = []
    depth_errors: list[float] = []
    uv_errors: list[float] = []
    radiance_errors: list[float] = []
    exact_solves = 0
    for li in range(0, len(leaves), cfg.validation_stride):
        leaf = leaves[li]
        probes_s, probes_local = _cell_probes(leaf.cell, scale, device=camera_center.device, dtype=camera_center.dtype)
        pred = _predict(leaf.anchor_s, leaf.anchor_u, leaf.j_us, probes_s)
        production = pred
        if cfg.newton_corrections:
            production, _ = newton_correct(reflector, surface, probes_s, camera_center, production)
        exact = solve_specular(reflector, surface, probes_s, camera_center, init_u=production)
        exact_solves += probes_s.shape[0]
        pp, _, pt = _project_and_tau(reflector, surface, camera, probes_s, production)
        ep, _, et = _project_and_tau(reflector, surface, camera, probes_s, exact.u)
        root_errors.extend(torch.linalg.vector_norm(pp - ep, dim=-1).detach().cpu().tolist())

        corners_s = _cell_corners(leaf.cell, scale, device=camera_center.device, dtype=camera_center.dtype)
        cpred = _predict(leaf.anchor_s, leaf.anchor_u, leaf.j_us, corners_s)
        cprod = cpred
        if cfg.newton_corrections:
            cprod, _ = newton_correct(reflector, surface, corners_s, camera_center, cprod)
        cp, _, ct = _project_and_tau(reflector, surface, camera, corners_s, cprod)
        ip = _triangle_interpolate_corners(cp, probes_local)
        probe_tri_p = _select_probe_triangles(cp, probes_local)
        probe_tri_tau = _select_probe_triangles(ct, probes_local)
        probe_tri_s = _select_probe_triangles(corners_s, probes_local)
        weights, _ = _screen_barycentric(probe_tri_p, ep)
        it = _weighted_triangle(probe_tri_tau, weights)
        isrc = _weighted_triangle(probe_tri_s, weights)
        screen_errors.extend(torch.linalg.vector_norm(ep - ip, dim=-1).detach().cpu().tolist())
        depth_errors.extend((et - it).abs().detach().cpu().tolist())
        uv_errors.extend(torch.linalg.vector_norm(probes_s - isrc, dim=-1).detach().cpu().tolist())
        radiance_errors.extend((surface.radiance(probes_s) - surface.radiance(isrc)).abs().amax(dim=-1).detach().cpu().tolist())

    out: dict[str, float | int] = {"validation_exact_solves": exact_solves}
    out.update(_quantiles(root_errors, "validated_root_screen_error_px"))
    out.update(_quantiles(screen_errors, "validated_triangle_screen_error_px"))
    out.update(_quantiles(depth_errors, "validated_triangle_depth_error"))
    out.update(_quantiles(uv_errors, "validated_source_uv_error"))
    out.update(_quantiles(radiance_errors, "validated_radiance_error"))
    return out



def build_adaptive_transport_mesh(
    reflector: HeightFieldReflector,
    surface: PlaneSurface,
    camera: PinholeCamera,
    cfg: AdaptiveTransportConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AdaptiveMeshResult:
    camera_center, _, _, _, _ = camera.matrices(device=device, dtype=dtype)
    base = cfg.base_tiles_per_axis
    # Two extra dyadic units are unnecessary; one extra bit guarantees that the
    # center of a max-depth cell is an integer grid key.
    base_cell_size = 1 << (cfg.max_depth + 1)
    scale = base * base_cell_size
    frontier = [
        _Cell(x0=ix * base_cell_size, y0=iy * base_cell_size, size=base_cell_size, depth=0)
        for iy in range(base)
        for ix in range(base)
    ]
    leaves: list[_Leaf] = []
    split_tiles = 0
    evaluated_tiles = 0
    tolerance_exhausted = 0
    rejected_singularity = 0
    root_errors: list[float] = []
    screen_errors: list[float] = []
    depth_errors: list[float] = []
    uv_errors: list[float] = []
    radiance_errors: list[float] = []
    kappas: list[float] = []
    sigmas: list[float] = []
    dets: list[float] = []

    while frontier:
        cell = frontier.pop()
        leaf = _evaluate_cell(reflector, surface, camera, camera_center, cell, scale, cfg)
        evaluated_tiles += 1
        root_errors.append(leaf.max_root_error_px)
        screen_errors.append(leaf.max_screen_interp_error_px)
        depth_errors.append(leaf.max_depth_interp_error)
        uv_errors.append(leaf.max_source_uv_error)
        radiance_errors.append(leaf.max_radiance_error)
        kappas.append(leaf.max_kappa_fu)
        sigmas.append(leaf.min_sigma_jphi)
        dets.append(leaf.min_abs_det_jphi)

        can_split = cell.depth < cfg.max_depth and cell.size >= 4
        if not leaf.tolerance_met and can_split:
            children = _child_cells(leaf)
            if len(leaves) + len(frontier) + len(children) > cfg.max_leaf_tiles:
                tolerance_exhausted += 1
                leaves.append(leaf)
            else:
                split_tiles += 1
                frontier.extend(children)
            continue
        if not leaf.tolerance_met:
            tolerance_exhausted += 1
            if not leaf.probe_valid or not leaf.orientation_consistent or leaf.min_sigma_jphi < cfg.min_sigma_jphi:
                rejected_singularity += 1
        leaves.append(leaf)

    vertex_keys, tri_list, incident, tri_owner = _build_conforming_topology(leaves)
    source_vertices = torch.stack(
        [_source_from_grid(x, y, scale, device=device, dtype=dtype) for x, y in vertex_keys], dim=0
    ) if vertex_keys else torch.empty((0, 2), device=device, dtype=dtype)

    reflector_u = torch.empty_like(source_vertices)
    emitted_valid = torch.zeros((source_vertices.shape[0],), device=device, dtype=torch.bool)
    for vi, s in enumerate(source_vertices):
        li = _choose_incident_leaf(leaves, incident[vi])
        leaf = leaves[li]
        pred = _predict(leaf.anchor_s, leaf.anchor_u, leaf.j_us, s.unsqueeze(0))
        prod = pred
        if cfg.newton_corrections:
            prod, _ = newton_correct(reflector, surface, s.unsqueeze(0), camera_center, prod)
        reflector_u[vi] = prod[0]
        emitted_valid[vi] = _physical_valid(reflector, surface, camera_center, s.unsqueeze(0), prod)[0]

    if source_vertices.numel():
        screen_vertices, z, tau_vertices = _project_and_tau(reflector, surface, camera, source_vertices, reflector_u)
        emitted_valid &= z > 0
    else:
        screen_vertices = torch.empty((0, 2), device=device, dtype=dtype)
        tau_vertices = torch.empty((0,), device=device, dtype=dtype)

    triangles = torch.tensor(tri_list, device=device, dtype=torch.int64) if tri_list else torch.empty((0, 3), device=device, dtype=torch.int64)
    triangle_valid = emitted_valid[triangles].all(dim=-1) if triangles.numel() else torch.empty((0,), device=device, dtype=torch.bool)
    if triangles.numel():
        tri_p = screen_vertices[triangles]
        area2 = (
            (tri_p[:, 1, 0] - tri_p[:, 0, 0]) * (tri_p[:, 2, 1] - tri_p[:, 0, 1])
            - (tri_p[:, 2, 0] - tri_p[:, 0, 0]) * (tri_p[:, 1, 1] - tri_p[:, 0, 1])
        )
        triangle_valid &= area2.abs() > 1.0e-8

    stats: dict[str, float | int] = {
        "adaptive_base_tiles": base * base,
        "adaptive_evaluated_tiles": evaluated_tiles,
        "adaptive_split_tiles": split_tiles,
        "adaptive_leaf_tiles": len(leaves),
        "adaptive_tolerance_exhausted_tiles": tolerance_exhausted,
        "adaptive_rejected_singularity_tiles": rejected_singularity,
        "adaptive_max_depth_reached": max((leaf.cell.depth for leaf in leaves), default=0),
        "exact_anchor_solves": evaluated_tiles,
        "adaptation_production_newton_corrections": evaluated_tiles * 9 * cfg.newton_corrections,
        "adaptation_estimator_newton_corrections": evaluated_tiles * 9,
        "emitted_vertex_newton_corrections": int(source_vertices.shape[0] * cfg.newton_corrections),
        "fixed_newton_corrections": (evaluated_tiles * 9 + int(source_vertices.shape[0])) * cfg.newton_corrections,
        "total_newton_linear_solves": evaluated_tiles * 9 * (1 + cfg.newton_corrections) + int(source_vertices.shape[0] * cfg.newton_corrections),
        "generated_vertices": int(source_vertices.shape[0]),
        "generated_triangles": int(triangle_valid.sum().item()) if triangle_valid.numel() else 0,
        "invalid_triangles": int((~triangle_valid).sum().item()) if triangle_valid.numel() else 0,
        "max_kappa_fu": max(kappas, default=float("nan")),
        "min_sigma_min_jphi": min(sigmas, default=float("nan")),
        "min_abs_det_jphi": min(dets, default=float("nan")),
    }
    stats.update(_quantiles(root_errors, "estimated_root_screen_error_px"))
    stats.update(_quantiles(screen_errors, "estimated_triangle_screen_error_px"))
    stats.update(_quantiles(depth_errors, "estimated_triangle_depth_error"))
    stats.update(_quantiles(uv_errors, "estimated_source_uv_error"))
    stats.update(_quantiles(radiance_errors, "estimated_radiance_error"))
    final_probe_stats = _final_triangle_probe_stats(
        reflector, surface, camera, camera_center, leaves, source_vertices,
        screen_vertices, tau_vertices, triangles, triangle_valid, tri_owner, cfg,
    )
    stats.update(final_probe_stats)
    stats["total_newton_linear_solves"] += int(final_probe_stats["final_topology_newton_linear_solves"])
    if cfg.validate_exact_probes:
        stats.update(_validate_exact_probes(reflector, surface, camera, camera_center, leaves, scale, cfg))
    else:
        stats["validation_exact_solves"] = 0

    return AdaptiveMeshResult(
        source_vertices=source_vertices,
        reflector_u=reflector_u,
        screen_vertices=screen_vertices,
        tau_vertices=tau_vertices,
        triangles=triangles,
        triangle_valid=triangle_valid,
        leaf_count=len(leaves),
        stats=stats,
    )
