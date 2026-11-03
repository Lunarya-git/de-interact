#!/usr/bin/env python
"""Stage 3: build + cache every split, run the leakage assertions, print/save the split statistics.

Changes vs. v1:
  - structure_groups.csv now contains chem_group (chemistry-only, no fp_hash), struct_group
    (legacy, includes fp_hash), and parent_group columns.
  - stale cached splits (built under a different leakage_unit) are detected and deleted.
  - entity_pools_seed42.json is written to data/processed/ for the tuning/evaluation protocol.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, pick_device, save_json
from deint.experiments import Context
from deint.splits import KINDS
from deint.structure import build_structure_groups, structure_report
from deint.entity_pool import build_entity_pools, save_entity_pools, assert_pool_independence


def purge_stale_splits(proc: Path, new_leakage_unit: str) -> int:
    """Delete cached splits whose filename indicates a different leakage unit."""
    splits_dir = proc / "splits"
    if not splits_dir.exists():
        return 0
    n = 0
    for f in splits_dir.glob("*.json"):
        parts = f.stem.split("__")
        if len(parts) >= 2:
            cached_unit = parts[1]
            if cached_unit != new_leakage_unit:
                f.unlink()
                n += 1
                print(f"  purged stale split: {f.name} (was {cached_unit}, now {new_leakage_unit})")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--pool-seed", type=int, default=42, help="Seed for entity-pool D/E draw")
    ap.add_argument("--pool-frac", type=float, default=0.25)
    ap.add_argument("--skip-splits", action="store_true", help="Only rebuild structure groups and pools")
    args = ap.parse_args()
    cfg = load_config(args.config)
    proc = resolve(cfg, "processed_dir")
    leakage_unit = cfg["split"].get("leakage_unit", "chem")

    # ── 3a. Purge stale splits ───────────────────────────────────────────────
    n_purged = purge_stale_splits(proc, leakage_unit)
    if n_purged:
        print(f"Purged {n_purged} stale cached splits (leakage_unit changed to '{leakage_unit}').")

    # ── 3b. Structure-identity groups ────────────────────────────────────────
    z = np.load(proc / "fingerprints.npz", allow_pickle=True)
    mols  = pd.read_csv(proc / "molecules.csv")
    pairs = pd.read_csv(proc / "pairs.csv")

    print(f"\nBuilding structure groups (leakage_unit='{leakage_unit}')...")
    sg = build_structure_groups(mols, z["fp"], z["cid"])
    sg.to_csv(proc / "structure_groups.csv", index=False)
    print(f"  Saved structure_groups.csv  ({len(sg)} rows, columns: {list(sg.columns)})")

    rep = structure_report(sg, pairs)
    save_json(rep, proc / "structure_audit.json")
    print("Structure report:")
    for grp in ("chem_group", "struct_group", "parent_group"):
        if grp in rep:
            r = rep[grp]
            print(f"  {grp}: {r['n_groups']} groups, "
                  f"{r['n_multi_member_groups']} multi-member, "
                  f"{r['group_pairs_with_conflicting_labels']} conflicting-label group-pairs")
    if "fp_collisions" in rep:
        fc = rep["fp_collisions"]
        print(f"  fp_collisions: {fc['fingerprints_shared_by_distinct_chem_groups']} fps shared "
              f"by {fc['chem_groups_involved']} distinct chem_groups")

    # ── 3c. Entity pools (D/E) for tuning/evaluation protocol ────────────────
    pools = build_entity_pools(mols, sg, pool_frac=args.pool_frac, seed=args.pool_seed)
    assert_pool_independence(pools)
    pool_path = save_entity_pools(pools, proc)
    print(f"\nEntity pools (seed={args.pool_seed}, frac={args.pool_frac}):")
    st = pools["stats"]
    print(f"  Pool E: {st['n_E_drug_groups']} drug-groups + {st['n_E_excipient_groups']} exc-groups "
          f"= {st['n_E_cids']} CIDs")
    print(f"  Pool D: {st['n_D_drug_groups']} drug-groups + {st['n_D_excipient_groups']} exc-groups "
          f"= {st['n_D_cids']} CIDs")
    print(f"  Saved {pool_path.name}")

    if args.skip_splits:
        print("\nSkipping split generation (--skip-splits).")
        return

    # ── 3d. Splits ───────────────────────────────────────────────────────────
    ctx = Context(cfg, pick_device("cpu"))
    rows = []
    seeds_to_run = args.seeds if args.seeds is not None else cfg["seeds"]
    for kind in KINDS:
        for seed in seeds_to_run:
            sp = ctx.get_split(kind, seed)   # assertions run inside make_split
            r  = sp["report"]
            rows.append({
                "split": kind, "seed": seed,
                **{f"{k}_n":   r[k]["n"]     for k in ("train", "val", "test")},
                **{f"{k}_pos": r[k]["n_pos"] for k in ("train", "val", "test")},
                "test_pair_seen": round(r["test"]["frac_pair_seen_in_train"], 3),
                "test_drug_seen": round(r["test"]["frac_drug_seen_in_train"], 3),
                "test_exc_seen":  round(r["test"]["frac_excipient_seen_in_train"], 3),
                "leakage_unit":   r.get("leakage_unit", "cid"),
            })

    df = pd.DataFrame(rows)
    out = proc / "split_summary.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 25):
        print("\nSplit summary (averaged over seeds):")
        print(df.groupby("split").mean(numeric_only=True).drop(columns="seed").round(3))
    print(f"\nAll leakage assertions passed.  Saved {out}")


if __name__ == "__main__":
    main()
