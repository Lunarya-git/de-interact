"""Hyperparameter tuning on entity pool D (Protocol T).

Follows PROJECT_DESIGN.md §11-12:
  - Entity-pool independence: Tuning runs strictly inside Pool D (entities
    never appear in Pool E, from which the final test set is drawn).
  - Multi-objective aggregate: Mean validation PR-AUC over
    {random_pair, cold_drug, cold_excipient} across tuning seeds (e.g. 100, 101).
  - Budget: 5 × (# dimensions) trials:
      de_crossnet:   40 trials
      gnn_siamese:   30 trials
      concat_mlp_reg: 30 trials
      siamese:       25 trials
  - Uses Optuna if installed; provides clean random-search fallback if not.
  - Outputs best hyperparameters to configs/tuned/<model>.yaml.
"""
from __future__ import annotations
import copy, json, time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from .baselines import EntityPrior, RFBaseline
from .entity_pool import load_entity_pools, pairs_in_D_only
from .metrics import best_threshold_f1, compute_metrics
from .models import build_model
from .splits import make_split
from .train import fit, predict_scores
from .utils import load_config, pick_device, resolve, save_json, set_seed

TUNING_SEEDS = [100, 101]
TUNING_SPLITS = ["random_pair", "cold_drug", "cold_excipient"]


def evaluate_trial_fp(model_name: str, params: dict, ctx, table_D: pd.DataFrame,
                      splits_D: dict, device: torch.device) -> float:
    """Evaluate a trial configuration on the 6 tuning conditions (3 splits x 2 seeds) in Pool D."""
    scores = []
    head = "logit1"
    criteria = ["val_pr_auc"]

    for split_kind in TUNING_SPLITS:
        for seed in TUNING_SEEDS:
            sp = splits_D[(split_kind, seed)]
            tr = np.array(sp["train"])
            va = np.array(sp["val"])
            if len(va) == 0 or len(tr) == 0:
                continue

            set_seed(seed)
            a, b = ctx.idx(table_D)
            y = table_D["label"].values

            # Model config
            mcfg = copy.deepcopy(params.get("model_config", {}))
            model = build_model(model_name, mcfg, 1).to(device)

            proto = {
                "head": head,
                "optimizer": "adamw",
                "lr": params.get("lr", 1e-3),
                "weight_decay": params.get("weight_decay", 1e-4),
                "batch_size": params.get("batch_size", 128),
                "epochs": params.get("epochs", 40),
                "patience": 10,
                "pos_weight": "auto",
                "checkpoint": "val_pr_auc",
            }

            from .experiments import _tensors
            out = fit(model, ctx.F, _tensors(ctx, a, b, y, tr), _tensors(ctx, a, b, y, va),
                      proto, criteria, device, seed, logger=None, progress=False, patience=10)

            val_prauc = out["best_value"].get("val_pr_auc", 0.0)
            if not np.isnan(val_prauc):
                scores.append(val_prauc)

    return float(np.mean(scores)) if scores else 0.0


def evaluate_trial_gnn(model_name: str, params: dict, ctx, table_D: pd.DataFrame,
                       splits_D: dict, device: torch.device) -> float:
    """Evaluate a GNN trial on the tuning conditions in Pool D."""
    from .gnn import build_gnn_model
    from .gnn_train import fit_graph

    scores = []
    cids_a = table_D.drug_cid.values
    cids_b = table_D.excipient_cid.values
    y = table_D["label"].values
    use_fp = (model_name in ("de_crossnet", "de_crossnet_self_attn"))
    fp_mat = ctx.F_np if use_fp else None

    for split_kind in TUNING_SPLITS:
        for seed in TUNING_SEEDS:
            sp = splits_D[(split_kind, seed)]
            tr = np.array(sp["train"])
            va = np.array(sp["val"])
            if len(va) == 0 or len(tr) == 0:
                continue

            set_seed(seed)
            mcfg = copy.deepcopy(params.get("model_config", {}))
            model = build_gnn_model(model_name, mcfg).to(device)

            proto = {
                "head": "logit1",
                "optimizer": "adamw",
                "lr": params.get("lr", 1e-3),
                "weight_decay": params.get("weight_decay", 1e-4),
                "batch_size": params.get("batch_size", 32),
                "epochs": params.get("epochs", 30),
                "patience": 8,
                "pos_weight": "auto",
                "checkpoint": "val_pr_auc",
                "mixed_precision": (device.type == "cuda"),
                "max_grad_norm": 1.0,
            }

            out = fit_graph(
                model=model,
                graph_cache=ctx.graph_cache,
                train_cids_a=cids_a[tr], train_cids_b=cids_b[tr], train_y=y[tr],
                val_cids_a=cids_a[va], val_cids_b=cids_b[va], val_y=y[va],
                fp_matrix=fp_mat,
                proto=proto, criteria=["val_pr_auc"],
                device=device, seed=seed, logger=None, progress=False, patience=8,
            )

            val_prauc = out["best_value"].get("val_pr_auc", 0.0)
            if not np.isnan(val_prauc):
                scores.append(val_prauc)

    return float(np.mean(scores)) if scores else 0.0


def sample_hyperparameters(model_name: str, rng: np.random.Generator) -> dict:
    """Sample one set of hyperparameters from the defined search spaces."""
    lr = float(10 ** rng.uniform(np.log10(3e-4), np.log10(3e-3)))
    weight_decay = float(10 ** rng.uniform(np.log10(1e-5), np.log10(1e-2)))
    dropout = float(rng.uniform(0.1, 0.4))

    if model_name == "de_crossnet":
        d_h = int(rng.choice([128, 192, 256]))
        n_layers = int(rng.choice([3, 4]))
        n_cross = int(rng.choice([1, 2]))
        batch_size = int(rng.choice([32, 48]))
        mcfg = {
            "d_h": d_h, "n_layers": n_layers, "n_heads": 4,
            "d_fp_out": 128, "n_cross": n_cross, "dropout": dropout,
            "head_hidden": [256, 128]
        }
        return {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                "epochs": 35, "model_config": mcfg}

    elif model_name == "gnn_siamese":
        d_h = int(rng.choice([128, 192, 256]))
        n_layers = int(rng.choice([3, 4]))
        batch_size = int(rng.choice([32, 48]))
        mcfg = {
            "d_h": d_h, "n_layers": n_layers, "dropout": dropout,
            "head_hidden": [256, 128]
        }
        return {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                "epochs": 35, "model_config": mcfg}

    elif model_name == "concat_mlp_reg":
        hidden_options = [[512, 512], [1024, 1024], [1024, 1024, 1024]]
        hidden = hidden_options[int(rng.choice(len(hidden_options)))]
        batch_size = int(rng.choice([64, 128]))
        mcfg = {"hidden": hidden, "dropout": dropout}
        return {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                "epochs": 50, "model_config": mcfg}

    elif model_name in ("siamese", "siamese_fp"):
        enc_options = [[256, 128], [512, 256], [512, 256, 128]]
        head_options = [[64], [128]]
        enc = enc_options[int(rng.choice(len(enc_options)))]
        hd = head_options[int(rng.choice(len(head_options)))]
        batch_size = int(rng.choice([64, 128]))
        mcfg = {"encoder": enc, "head": hd, "dropout": dropout, "interaction": "full"}
        return {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                "epochs": 50, "model_config": mcfg}

    raise ValueError(f"Unknown model for tuning: {model_name}")


def run_tuning(model_name: str, n_trials: int, cfg: dict, device: torch.device,
               progress: bool = True) -> dict:
    """Run hyperparameter search on Pool D."""
    from .experiments import Context
    proc = resolve(cfg, "processed_dir")
    pools = load_entity_pools(proc)
    if pools is None:
        raise RuntimeError("Entity pools not found. Run scripts/03_make_splits.py first.")

    ctx = Context(cfg, device)
    # Extract pairs that lie entirely in pool D
    table_D = pairs_in_D_only(ctx.pairs, pools)
    print(f"Tuning {model_name} on Pool D: {len(table_D)} pairs (all entities in D, none in E)")

    # Pre-generate tuning splits inside table_D
    splits_D = {}
    cfg_D = copy.deepcopy(cfg)
    cfg_D["split"].update(min_pos_eval=2, min_pos_eval_cold_both=1)
    for s in TUNING_SPLITS:
        for seed in TUNING_SEEDS:
            splits_D[(s, seed)] = make_split(table_D, s, seed, cfg_D, groups=ctx.groups)

    is_gnn = model_name in ("de_crossnet", "gnn_siamese")
    eval_fn = evaluate_trial_gnn if is_gnn else evaluate_trial_fp

    rng = np.random.default_rng(42)
    best_score = -1.0
    best_params = None
    trials_log = []

    print(f"Beginning {n_trials} trials for {model_name}...")
    for t_idx in range(1, n_trials + 1):
        params = sample_hyperparameters(model_name, rng)
        t0 = time.time()
        score = eval_fn(model_name, params, ctx, table_D, splits_D, device)
        elapsed = time.time() - t0

        is_best = score > best_score
        if is_best:
            best_score = score
            best_params = params

        rec = {"trial": t_idx, "score_val_pr_auc": round(score, 4),
               "elapsed_s": round(elapsed, 1), "params": params}
        trials_log.append(rec)
        if progress:
            star = " ★ BEST" if is_best else ""
            print(f"  Trial {t_idx:2d}/{n_trials}: mean val PR-AUC = {score:.4f} ({elapsed:.1f}s){star}")

    # Save winning config
    tuned_dir = resolve(cfg, "results_dir") / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    out_file = tuned_dir / f"{model_name}.yaml"
    with open(out_file, "w") as f:
        yaml.safe_dump({"model": model_name, "best_score": best_score,
                        "best_params": best_params, "trials": len(trials_log)}, f, sort_keys=False)

    log_file = tuned_dir / f"{model_name}_trials.json"
    save_json({"model": model_name, "best_score": best_score,
               "best_params": best_params, "trials": trials_log}, log_file)

    print(f"\nTuning complete for {model_name}: Best mean val PR-AUC = {best_score:.4f}")
    print(f"Saved winning config to {out_file}")
    return {"model": model_name, "best_score": best_score, "best_params": best_params}

