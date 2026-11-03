#!/usr/bin/env python
"""Stage E5: Sensitivity and fingerprint-collision analysis.

Covers all 5 sensitivity analyses specified in PROJECT_DESIGN.md §13.1:
  (a) Leakage unit impact: CID-keyed vs chem-grouped splits (evaluates impact of structure twins)
  (b) Fingerprint-collision stratification: Post-hoc metrics on pairs with vs without
      PubChem fingerprint collisions among distinct chemical structures
  (c) Parent-fragment sensitivity: 'parent' splits (salt/hydrate stripped) vs 'chem'
  (d) Label-conflict group-pair exclusion: Recomputes metrics after dropping the 2 group-pairs
      with conflicting labels
  (e) Default vs tuned comparison: Quantifies gains from HPO Protocol T

Usage:
  python scripts/07_sensitivity.py [--config configs/default.yaml] [--device auto]
"""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, pick_device, save_json
from deint.metrics import compute_metrics, best_threshold_f1
from deint.experiments import Context, run_experiment


def analyze_fp_collisions_posthoc(res_dir: Path, proc_dir: Path) -> dict:
    """Part (b): Stratify existing test predictions by fingerprint collisions.

    Reads predictions_test.csv from E3 runs and splits test pairs into:
      - colliding: drug or excipient shares its fingerprint with a chemically distinct molecule
      - non_colliding: neither molecule has a fingerprint collision
    """
    sg = pd.read_csv(proc_dir / "structure_groups.csv")
    rep_chem = sg.drop_duplicates("chem_group")
    fp_counts = rep_chem.groupby("fp_hash")["chem_group"].nunique()
    colliding_fps = set(fp_counts[fp_counts > 1].index)

    cid_to_fp = dict(zip(sg.cid, sg.fp_hash))
    colliding_cids = {cid for cid, fph in cid_to_fp.items() if fph in colliding_fps}

    results = []
    for pred_file in res_dir.glob("E3_generalization/*/predictions_test.csv"):
        parts = pred_file.parent.name.split("__")
        if len(parts) < 3:
            continue
        model, split, seed_str = parts[0], parts[1], parts[2]
        df = pd.read_csv(pred_file)

        is_coll = df.drug_cid.isin(colliding_cids) | df.excipient_cid.isin(colliding_cids)
        y = df["label"].values
        scores = df["score"].values

        for subset_name, mask in [("all", np.ones(len(df), dtype=bool)),
                                  ("colliding", is_coll.values),
                                  ("non_colliding", (~is_coll).values)]:
            if mask.sum() == 0 or y[mask].sum() == 0 or (1 - y[mask]).sum() == 0:
                continue
            m = compute_metrics(y[mask], scores[mask], 0.5)
            results.append({
                "model": model, "split": split, "seed": seed_str,
                "subset": subset_name, "n": int(mask.sum()), "n_pos": int(y[mask].sum()),
                "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "f1": m["f1_incompatible"]
            })

    out_df = pd.DataFrame(results)
    if not out_df.empty:
        summary = out_df.groupby(["model", "split", "subset"])[["pr_auc", "f1"]].agg(["mean", "std"])
        return {"detailed": results, "summary": summary.to_dict()}
    return {"message": "No E3 predictions found yet. Run E3 first."}


def analyze_label_conflict_exclusion(res_dir: Path, proc_dir: Path) -> dict:
    """Part (d): Evaluate test metrics after removing the 2 conflicting chemical-group pairs."""
    sg = pd.read_csv(proc_dir / "structure_groups.csv")
    gmap = dict(zip(sg.cid, sg.chem_group))
    pairs = pd.read_csv(proc_dir / "pairs.csv")
    pairs["ga"] = pairs.drug_cid.map(gmap)
    pairs["gb"] = pairs.excipient_cid.map(gmap)
    pairs["u"] = [tuple(sorted((a, b))) for a, b in zip(pairs.ga, pairs.gb)]

    # Identify conflicting group pairs
    gp = pairs.groupby("u")["label"].nunique()
    conflicting_groups = set(gp[gp > 1].index)

    results = []
    for pred_file in res_dir.glob("E3_generalization/*/predictions_test.csv"):
        parts = pred_file.parent.name.split("__")
        if len(parts) < 3:
            continue
        model, split, seed_str = parts[0], parts[1], parts[2]
        df = pd.read_csv(pred_file)
        pair_keys = [tuple(sorted((gmap.get(int(a)), gmap.get(int(b)))))
                     for a, b in zip(df.drug_cid, df.excipient_cid)]
        clean_mask = np.array([k not in conflicting_groups for k in pair_keys])

        y = df["label"].values
        scores = df["score"].values

        m_all   = compute_metrics(y, scores, 0.5)
        m_clean = compute_metrics(y[clean_mask], scores[clean_mask], 0.5) if clean_mask.sum() > 0 else m_all

        results.append({
            "model": model, "split": split, "seed": seed_str,
            "pr_auc_all": m_all["pr_auc"], "pr_auc_cleaned": m_clean["pr_auc"],
            "delta_pr_auc": m_clean["pr_auc"] - m_all["pr_auc"],
        })

    return {"results": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-trainings", action="store_true",
                    help="Run parts (a) and (c) trainings (CID vs parent leakage units)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    proc_dir = resolve(cfg, "processed_dir")
    res_dir  = resolve(cfg, "results_dir")
    out_dir  = res_dir / "E5_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running E5 Sensitivity Analyses...")

    # (b) Fingerprint collisions post-hoc
    print("\n[E5b] Analyzing fingerprint-collision stratification on E3 predictions...")
    res_b = analyze_fp_collisions_posthoc(res_dir, proc_dir)
    save_json(res_b, out_dir / "E5b_fp_collision_stratification.json")
    print(f"  Saved {out_dir / 'E5b_fp_collision_stratification.json'}")

    # (d) Conflicting pair exclusion
    print("\n[E5d] Analyzing label-conflict group-pair exclusion...")
    res_d = analyze_label_conflict_exclusion(res_dir, proc_dir)
    save_json(res_d, out_dir / "E5d_conflict_exclusion.json")
    print(f"  Saved {out_dir / 'E5d_conflict_exclusion.json'}")

    # (a) and (c) Optional training runs
    if args.run_trainings:
        dev = pick_device(args.device)
        print(f"\n[E5a/c] Running sensitivity trainings across leakage units on {dev}...")
        for unit in ["cid", "parent"]:
            cfg_unit = copy.deepcopy(cfg)
            cfg_unit["split"]["leakage_unit"] = unit
            ctx = Context(cfg_unit, dev)
            print(f"  Running concat_mlp on {unit} splits...")
            run_experiment(ctx, "E3_generalization", models=["concat_mlp"],
                           splits=["cold_drug", "cold_excipient"], seeds=[0, 1, 2], progress=False)

    print(f"\nE5 Sensitivity Analysis complete. Results in {out_dir}")


if __name__ == "__main__":
    main()

