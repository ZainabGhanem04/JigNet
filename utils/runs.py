"""Run directories: every experiment writes its config copy + environment info next to its results."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import save_config


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "git_commit": _git_commit()}
    for mod in ("torch", "torchvision", "numpy", "pandas", "pulp", "highspy", "cv2"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info


def make_run_dir(base_dir: str, name: str, cfg) -> Path:
    """Create <base_dir>/<name>/ (re-used if it exists so training can resume) and store config + env."""
    run_dir = Path(base_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    env = environment_info()
    env["created"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    return run_dir
