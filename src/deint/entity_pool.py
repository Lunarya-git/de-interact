"""Entity pool construction for the independent tuning/evaluation protocol.

Overview
--------
To ensure that hyperparameter tuning cannot exploit information from the final
test molecules, the dataset is partitioned into two entity pools at the
*chemistry-group* level, BEFORE any split:

  Pool E (evaluation-only): 25 % of drug chem-groups + 25 % of excipient
         chem-groups (label-blind draw, fixed seed).  Molecules that appear
         in both drug and excipient roles are always assigned to pool D
         (they cannot be fully held-out without discarding too many pairs).

  Pool D (development): everything else.  Tuning runs inside D only.

Final test pairs are drawn exclusively from E (cold_drug / cold_excipient /
cold_both relative to E-only molecules).  The model trained on all of D is
evaluated on E.

Guarantees
----------
  - Pool assignment is entirely label-blind and uses only chem_groups.
  - assert_pool_independence() verifies that no E entity ever appears in D.
  - cached on disk in data/processed/entity_pools_seed{seed}.json so that
    all experiments use the same fixed draw.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


# ── construction ─────────────────────────────────────────────────────────────

def build_entity_pools(
    molecules_df: pd.DataFrame,
    struct_groups_df: pd.DataFrame,
    pool_frac: float = 0.25,
    seed: int = 42,
) -> dict:
    """Create the D/E entity pools.

    Args:
        molecules_df:    molecules.csv (columns: cid, roles)
        struct_groups_df: structure_groups.csv (columns: cid, chem_group)
        pool_frac:       fraction of drug-only and excipient-only chem_groups
                         assigned to pool E (default 0.25).
        seed:            random seed for the label-blind draw.

    Returns a dict:
        {
          "seed": int,
          "pool_frac": float,
          "E_drug_groups":     list[int],   chem_group ids → pool E (drug role)
          "E_excipient_groups": list[int],
          "D_drug_groups":     list[int],
          "D_excipient_groups": list[int],
          "both_role_groups":  list[int],  always D
          "E_cids":            list[int],  ALL cids whose chem_group is in E
          "D_cids":            list[int],
          "stats": {...}
        }
    """
    role_map = dict(zip(molecules_df.cid, molecules_df.roles))
    cg_map   = dict(zip(struct_groups_df.cid, struct_groups_df.chem_group))

    # Assign each chem_group a role: drug | excipient | both
    group_roles: dict[int, set] = {}
    for cid, grp in cg_map.items():
        role = role_map.get(int(cid), "drug")
        if isinstance(role, str) and "+" in role:
            parts = set(role.split("+"))
        else:
            parts = {role}
        if grp not in group_roles:
            group_roles[grp] = set()
        group_roles[grp] |= parts

    drug_only_groups = sorted(g for g, r in group_roles.items() if r == {"drug"})
    exc_only_groups  = sorted(g for g, r in group_roles.items() if r == {"excipient"})
    both_groups      = sorted(g for g, r in group_roles.items() if len(r) > 1 or r == {"drug+excipient"})

    rng = np.random.default_rng(seed)
    n_drug_E = max(1, round(pool_frac * len(drug_only_groups)))
    n_exc_E  = max(1, round(pool_frac * len(exc_only_groups)))

    E_drug = sorted(rng.choice(drug_only_groups, n_drug_E, replace=False).tolist())
    E_exc  = sorted(rng.choice(exc_only_groups,  n_exc_E,  replace=False).tolist())

    E_set = set(E_drug) | set(E_exc)
    D_drug = [g for g in drug_only_groups if g not in E_set]
    D_exc  = [g for g in exc_only_groups  if g not in E_set]

    E_cids = sorted(cid for cid, grp in cg_map.items() if grp in E_set)
    D_cids = sorted(cid for cid, grp in cg_map.items() if grp not in E_set)

    return {
        "seed": seed,
        "pool_frac": pool_frac,
        "E_drug_groups":      E_drug,
        "E_excipient_groups": E_exc,
        "D_drug_groups":      D_drug,
        "D_excipient_groups": D_exc,
        "both_role_groups":   both_groups,
        "E_cids":             E_cids,
        "D_cids":             D_cids,
        "stats": {
            "n_drug_only_groups":      len(drug_only_groups),
            "n_excipient_only_groups": len(exc_only_groups),
            "n_both_role_groups":      len(both_groups),
            "n_E_drug_groups":         len(E_drug),
            "n_E_excipient_groups":    len(E_exc),
            "n_D_drug_groups":         len(D_drug),
            "n_D_excipient_groups":    len(D_exc),
            "n_E_cids":                len(E_cids),
            "n_D_cids":                len(D_cids),
        },
    }


# ── persistence ──────────────────────────────────────────────────────────────

def save_entity_pools(pools: dict, proc_dir: Path) -> Path:
    path = proc_dir / f"entity_pools_seed{pools['seed']}.json"
    path.write_text(json.dumps(pools, indent=2))
    return path


def load_entity_pools(proc_dir: Path, seed: int = 42) -> dict | None:
    path = proc_dir / f"entity_pools_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    path_default = proc_dir / "entity_pools.json"
    if path_default.exists():
        return json.loads(path_default.read_text())
    return None


# ── validation ───────────────────────────────────────────────────────────────

def assert_pool_independence(pools: dict) -> None:
    """Assert that E and D cid sets are disjoint."""
    E = set(pools["E_cids"]); D = set(pools["D_cids"])
    overlap = E & D
    assert not overlap, f"Pool E and D overlap on {len(overlap)} CIDs: {sorted(overlap)[:10]}"
    E_grp = set(pools["E_drug_groups"]) | set(pools["E_excipient_groups"])
    D_grp = set(pools["D_drug_groups"]) | set(pools["D_excipient_groups"])
    grp_overlap = E_grp & D_grp
    assert not grp_overlap, f"Pool E and D overlap on {len(grp_overlap)} chem_groups"


# ── pair filtering ────────────────────────────────────────────────────────────

def filter_pairs_to_pool(pairs_df: pd.DataFrame, pool_cids: list[int]) -> pd.DataFrame:
    """Return only pairs where BOTH drug and excipient are in the given cid set."""
    cid_set = set(pool_cids)
    mask = (pairs_df.drug_cid.isin(cid_set) & pairs_df.excipient_cid.isin(cid_set))
    return pairs_df[mask].reset_index(drop=True)


def pairs_in_D_only(pairs_df: pd.DataFrame, pools: dict) -> pd.DataFrame:
    """Return pairs where both molecules are in pool D (for tuning)."""
    return filter_pairs_to_pool(pairs_df, pools["D_cids"])


def pairs_in_E_only(pairs_df: pd.DataFrame, pools: dict) -> pd.DataFrame:
    """Return pairs where both molecules are in pool E (for final evaluation)."""
    return filter_pairs_to_pool(pairs_df, pools["E_cids"])

