#!/usr/bin/env python
"""Stage T: Hyperparameter tuning inside Entity Pool D (Protocol T).

Evaluates candidates on Pool D only (guaranteeing that evaluation Pool E
and the final test sets remain unobserved).

Usage:
  python scripts/06_tune.py --model de_crossnet --trials 10 [--device cuda]
  python scripts/06_tune.py --all-models [--device cuda]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, pick_device
from deint.tune import run_tuning

DEFAULT_BUDGETS = {
    "de_crossnet": 40,
    "gnn_siamese": 30,
    "concat_mlp_reg": 30,
    "siamese": 25,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["de_crossnet", "gnn_siamese", "concat_mlp_reg", "siamese"],
                    default=None, help="Model to tune")
    ap.add_argument("--all-models", action="store_true", help="Tune all 4 models in sequence")
    ap.add_argument("--trials", type=int, default=None, help="Override trial budget")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = pick_device(args.device)
    print(f"Device: {dev}")

    if args.all_models:
        models = ["de_crossnet", "gnn_siamese", "concat_mlp_reg", "siamese"]
    elif args.model:
        models = [args.model]
    else:
        ap.error("Specify --model <name> or --all-models")

    for m in models:
        budget = args.trials if args.trials is not None else DEFAULT_BUDGETS.get(m, 20)
        print(f"\n==========================================")
        print(f"Tuning {m} (budget: {budget} trials)")
        print(f"==========================================")
        run_tuning(m, budget, cfg, dev)


if __name__ == "__main__":
    main()

