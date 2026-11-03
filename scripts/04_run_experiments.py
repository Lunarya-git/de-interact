#!/usr/bin/env python
"""Stage 4: run experiments.  Completed runs are skipped (resume); --force archives and reruns."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, pick_device
from deint.experiments import Context, run_experiment


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", "-e", nargs="+", required=True,
                    help="E1_reproduction E2_leakage_audit E3_generalization E4_ablation, or 'all'")
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    ap.add_argument("--splits", nargs="*", default=None, help="restrict to these split kinds")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seed list")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--force", action="store_true", help="archive finished runs and redo them")
    ap.add_argument("--no-progress", action="store_true", help="hide per-epoch bars")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = pick_device(a.device)
    print(f"device: {dev}")
    ctx = Context(cfg, dev)
    exps = list(cfg["experiments"]) if a.experiment == ["all"] else a.experiment
    for e in exps:
        run_experiment(ctx, e, a.models, a.splits, a.seeds, a.force, not a.no_progress)


if __name__ == "__main__":
    main()
