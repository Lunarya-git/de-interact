#!/usr/bin/env python
"""Stage 3: build + cache every split, run the leakage assertions, print/save the split statistics."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, pick_device
from deint.experiments import Context
from deint.splits import KINDS
from deint.structure import build_structure_groups, structure_report
from deint.utils import save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None); ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); proc = resolve(cfg, "processed_dir")
    # 3a. structure-identity groups (needs the cached fingerprints from stage 2; labels are used for reporting only)
    z = np.load(proc / "fingerprints.npz", allow_pickle=True)
    mols = pd.read_csv(proc / "molecules.csv"); pairs = pd.read_csv(proc / "pairs.csv")
    sg = build_structure_groups(mols, z["fp"], z["cid"]); sg.to_csv(proc / "structure_groups.csv", index=False)
    rep = structure_report(sg, pairs); save_json(rep, proc / "structure_audit.json")
    print("Structure groups:", {k: v for k, v in rep.items()})
    # 3b. splits
    ctx = Context(cfg, pick_device("cpu"))
    rows = []
    for kind in KINDS:
        for seed in (args.seeds if args.seeds is not None else cfg["seeds"]):
            sp = ctx.get_split(kind, seed); r = sp["report"]       # assertions run inside make_split
            rows.append({"split": kind, "seed": seed, **{f"{k}_n": r[k]["n"] for k in ("train", "val", "test")},
                         **{f"{k}_pos": r[k]["n_pos"] for k in ("train", "val", "test")},
                         "test_pair_seen": round(r["test"]["frac_pair_seen_in_train"], 3),
                         "test_drug_seen": round(r["test"]["frac_drug_seen_in_train"], 3),
                         "test_exc_seen": round(r["test"]["frac_excipient_seen_in_train"], 3)})
    df = pd.DataFrame(rows); out = resolve(cfg, "processed_dir") / "split_summary.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.groupby("split").mean(numeric_only=True).drop(columns="seed").round(3))
    print(f"\nAll leakage assertions passed. Saved {out}")


if __name__ == "__main__":
    main()
