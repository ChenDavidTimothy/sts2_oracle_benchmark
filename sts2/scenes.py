from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

from .camera import PinholeCamera
from .geometry import HeightFieldReflector, PlaneSurface


@dataclass
class Scene:
    reflector: HeightFieldReflector
    camera: PinholeCamera
    surfaces: list[PlaneSurface]
    environment_rgb: tuple[float, float, float]


def scene_from_config(cfg: dict[str, Any]) -> Scene:
    rcfg = cfg["reflector"]
    reflector = HeightFieldReflector(**rcfg)
    ccfg = cfg["camera"]
    camera = PinholeCamera(**ccfg)
    surfaces = [PlaneSurface(**s) for s in cfg["surfaces"]]
    env = tuple(cfg.get("environment_rgb", [0.03, 0.04, 0.06]))
    return Scene(reflector=reflector, camera=camera, surfaces=surfaces, environment_rgb=env)


def _set_path(d: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = cur[p]
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


def expand_sweep(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = cfg.get("sweep", {})
    if not sweep:
        return [cfg]
    keys = list(sweep.keys())
    runs = []
    for values in product(*[sweep[k] for k in keys]):
        c = deepcopy(cfg)
        c.pop("sweep", None)
        tags = []
        for k, v in zip(keys, values):
            _set_path(c, k, v)
            tags.append(f"{k}={v}")
        c["_sweep_tag"] = ",".join(tags)
        runs.append(c)
    return runs
