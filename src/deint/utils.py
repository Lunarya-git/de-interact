"""Small shared helpers: config loading, seeding, timestamps, run directories, logging."""
from __future__ import annotations
import json, logging, os, random, shutil, time
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a config path relative to the repo root (portable across machines)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(pref: str = "auto"):
    import torch
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def prepare_run_dir(run_dir: Path, force: bool) -> bool:
    """Return True if the run should execute.  Completed runs are skipped (resume);
    with force=True an existing directory is ARCHIVED (never overwritten)."""
    done = run_dir / "metrics.json"
    if run_dir.exists() and done.exists():
        if not force:
            return False
        archive = run_dir.parent.parent / "_archive"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(archive / f"{run_dir.name}__{now_id()}"))
    elif run_dir.exists():                       # interrupted run: restart cleanly
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return True


def file_logger(path: Path, name: str) -> logging.Logger:
    """File-only logger (the console stays clean; tqdm shows progress)."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    h = logging.FileHandler(path, mode="a")
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)
    return logger


class Timer:
    def __init__(self): self.t0 = time.time()
    def elapsed(self) -> float: return time.time() - self.t0
    def fmt(self) -> str:
        s = int(self.elapsed()); return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"
