"""Configuration loading helpers (YAML -> dict) with project-relative path resolution."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into a deep copy of ``base``.

    Args:
        base: Original dictionary.
        overrides: Values that take precedence.

    Returns:
        A new merged dictionary.
    """
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load ``config.yaml``.

    Args:
        path: Optional path to a YAML file (defaults to the project's config.yaml).
        overrides: Optional nested dict merged on top of the file contents.

    Returns:
        The configuration dictionary.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate_config(cfg)
    return cfg


def resolve_path(p: str) -> Path:
    """Resolve a (possibly relative) config path against the project root.

    Args:
        p: Path string from the config.

    Returns:
        Absolute :class:`pathlib.Path`.
    """
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def smoke_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a tiny CPU-friendly config for smoke tests (synthetic data, no downloads).

    Args:
        overrides: Extra nested overrides applied last.

    Returns:
        Configuration dictionary with sizes shrunk for fast tests.
    """
    cfg = load_config()
    s = cfg["smoke"]
    small = {
        "data": {"image_size": s["image_size"], "num_workers": 0},
        "model": {"pretrained": s["pretrained"]},
        "student": {"pretrained": s["pretrained"]},
        "train": {"epochs": s["epochs"], "batch_size": s["batch_size"], "amp": False, "log_every": 1},
        "distill": {"epochs": 1},
        "loss": {"evidential_kl_anneal_epochs": 2},
        "paths": {"output_dir": "./outputs/smoke", "checkpoint_dir": "./outputs/smoke/checkpoints",
                  "report_dir": "./outputs/smoke/reports"},
        "inference": {"decision": {"slow_speed_floor": None}, "hood_polygon": None},
    }
    cfg = deep_update(cfg, small)
    return deep_update(cfg, overrides) if overrides else cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Reject invalid inference contracts before loading models or opening outputs."""
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    try:
        for section in ("paths", "data", "inference", "fusion", "camera"):
            if not isinstance(cfg[section], dict):
                raise ValueError(f"Config section {section} must be a mapping")
        for size in (cfg["data"]["image_size"], cfg["inference"]["input_size"]):
            if len(size) != 2 or any(not isinstance(n, int) or n <= 0 for n in size):
                raise ValueError("Image sizes must contain two positive integers (H, W)")
        for n in cfg["data"]["imagenet_std"]:
            if not math.isfinite(n) or n <= 0:
                raise ValueError("Normalization std must be finite and positive")
        for key in ("fallback_threshold", "mask_threshold"):
            value = cfg["fusion"][key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"fusion.{key} must be in [0, 1]")
        for key in ("max_steer_deg", "v_max", "max_speed_ramp", "max_steer_rate_deg", "speed_angle_denom"):
            value = cfg["inference"]["decision"][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"decision.{key} must be finite and positive")
        for key in ("queue_size", "frame_skip"):
            if not isinstance(cfg["inference"][key], int) or cfg["inference"][key] <= 0:
                raise ValueError(f"inference.{key} must be a positive integer")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid or missing configuration key: {exc}") from exc
