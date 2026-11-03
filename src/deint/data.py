"""Raw-data audit and deterministic cleaning.

The raw CSV is never modified.  Cleaning produces:
  pairs.csv      one row per unique (drug_cid, excipient_cid): label, multiplicity in raw file
  molecules.csv  one row per unique molecule CID: smiles, roles, RDKit validity
  raw_labeled.csv  the raw rows with a binary label (needed for the paper-faithful, leaky split)
  audit.json / audit.md  numbers quoted in the report
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

LABEL_MAP = {"Compatible": 0, "Imcompatible": 1}      # NOTE: the raw file misspells "Incompatible"
PAPER_REPORTED = {"n_incompatible": 179, "n_compatible": 3549, "n_nodes": 739, "n_edges": 3728}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"drug_cid", "excipient_cid", "drug_smiles", "excipient_smiles", "output_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing columns: {missing}")
    return df


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    unknown = set(df["output_value"].unique()) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unexpected label strings: {unknown}")
    out = df.copy()
    out["label"] = out["output_value"].map(LABEL_MAP).astype(int)
    return out


def rdkit_valid(smiles: str) -> bool:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles) is not None


def build_molecule_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per CID regardless of role.  Asserts CID -> SMILES is a function."""
    a = df[["drug_cid", "drug_smiles"]].set_axis(["cid", "smiles"], axis=1).assign(role="drug")
    b = df[["excipient_cid", "excipient_smiles"]].set_axis(["cid", "smiles"], axis=1).assign(role="excipient")
    long = pd.concat([a, b], ignore_index=True)
    n_smiles = long.groupby("cid")["smiles"].nunique()
    if (n_smiles > 1).any():
        raise ValueError(f"CID maps to >1 SMILES for {int((n_smiles > 1).sum())} CIDs")
    mol = long.groupby("cid").agg(smiles=("smiles", "first"),
                                  roles=("role", lambda r: "+".join(sorted(set(r))))).reset_index()
    mol["rdkit_valid"] = mol["smiles"].map(rdkit_valid)
    mol["n_components"] = mol["smiles"].str.count(r"\.") + 1
    return mol.sort_values("cid").reset_index(drop=True)


def clean(df_raw: pd.DataFrame):
    """Return (pairs, molecules, raw_labeled).  Strategy documented in PROJECT_DESIGN.md Sec. 4."""
    df = add_label(df_raw)
    key = ["drug_cid", "excipient_cid"]
    lab = df.groupby(key)["label"].nunique()
    if (lab > 1).any():                                   # none in this dataset; guard for reuse
        raise ValueError(f"{int((lab > 1).sum())} pairs have conflicting labels - resolve before use")
    pairs = (df.groupby(key, as_index=False)
               .agg(label=("label", "first"), n_raw_rows=("label", "size")))
    molecules = build_molecule_table(df)
    return pairs, molecules, df


def audit(df_raw: pd.DataFrame, pairs: pd.DataFrame, molecules: pd.DataFrame) -> dict:
    df = add_label(df_raw)
    key = ["drug_cid", "excipient_cid"]
    a = {}
    a["raw_rows"] = int(len(df))
    a["raw_columns"] = list(df_raw.columns)
    a["missing_values"] = {c: int(v) for c, v in df_raw.isna().sum().items()}
    a["raw_label_strings"] = df_raw["output_value"].value_counts().to_dict()
    a["exact_duplicate_rows"] = int(df_raw.duplicated().sum())
    a["unique_pairs"] = int(len(pairs))
    a["conflicting_label_pairs"] = int((df.groupby(key)["label"].nunique() > 1).sum())
    a["raw_incompatible_rows"] = int(df["label"].sum())
    a["raw_incompatible_fraction"] = float(df["label"].mean())
    a["clean_pairs_incompatible"] = int(pairs["label"].sum())
    a["clean_pairs_compatible"] = int((pairs["label"] == 0).sum())
    a["clean_incompatible_fraction"] = float(pairs["label"].mean())
    a["pair_multiplicity_counts"] = {int(k): int(v) for k, v in pairs["n_raw_rows"].value_counts().sort_index().items()}
    a["multiplicity_by_label"] = {int(l): pairs[pairs.label == l]["n_raw_rows"].describe().to_dict() for l in (0, 1)}
    a["unique_drugs"] = int(pairs["drug_cid"].nunique())
    a["unique_excipients"] = int(pairs["excipient_cid"].nunique())
    a["molecules_total"] = int(len(molecules))
    a["molecules_in_both_roles"] = int((molecules["roles"] == "drug+excipient").sum())
    a["swapped_role_duplicate_pairs"] = int(len(pairs) - pairs.assign(
        u=pairs.apply(lambda r: tuple(sorted((r.drug_cid, r.excipient_cid))), axis=1))["u"].nunique())
    a["rdkit_invalid_smiles"] = molecules.loc[~molecules.rdkit_valid, ["cid", "smiles"]].to_dict("records")
    a["multi_component_smiles(salts/mixtures)"] = int((molecules["n_components"] > 1).sum())
    sm = molecules.groupby("smiles")["cid"].nunique()
    a["smiles_shared_by_multiple_cids"] = {s: molecules[molecules.smiles == s].cid.tolist() for s in sm[sm > 1].index}
    # entity-level label concentration: the shortcut that motivates cold-start evaluation
    for role, col in [("drug", "drug_cid"), ("excipient", "excipient_cid")]:
        g = pairs.groupby(col)["label"].agg(["size", "mean"])
        a[f"{role}_label_concentration"] = {
            "entities": int(len(g)),
            "always_incompatible": int((g["mean"] == 1).sum()),
            "always_compatible": int((g["mean"] == 0).sum()),
            "mixed": int(((g["mean"] > 0) & (g["mean"] < 1)).sum()),
            "median_pairs_per_entity": float(g["size"].median()),
            "max_pairs_per_entity": int(g["size"].max()),
        }
    a["paper_reported"] = PAPER_REPORTED
    a["discrepancies_vs_paper"] = {
        "paper_incompatible_vs_ours_unique": [PAPER_REPORTED["n_incompatible"], a["clean_pairs_incompatible"]],
        "paper_compatible_vs_ours_unique": [PAPER_REPORTED["n_compatible"], a["clean_pairs_compatible"]],
        "paper_total_vs_ours_unique": [PAPER_REPORTED["n_incompatible"] + PAPER_REPORTED["n_compatible"], a["unique_pairs"]],
        "paper_nodes_vs_ours_molecules": [PAPER_REPORTED["n_nodes"], a["molecules_total"]],
        "paper_edges_vs_ours_unique_pairs": [PAPER_REPORTED["n_edges"], a["unique_pairs"]],
    }
    return a


def audit_markdown(a: dict) -> str:
    L = ["# Data audit", ""]
    L += [f"- Raw rows: **{a['raw_rows']}**; exact duplicate rows: **{a['exact_duplicate_rows']}**",
          f"- Unique (drug, excipient) pairs: **{a['unique_pairs']}** "
          f"({a['clean_pairs_compatible']} compatible / {a['clean_pairs_incompatible']} incompatible, "
          f"{100*a['clean_incompatible_fraction']:.2f}% incompatible)",
          f"- Raw incompatible fraction (row level): {100*a['raw_incompatible_fraction']:.2f}%",
          f"- Conflicting-label pairs: {a['conflicting_label_pairs']}; missing values: {sum(a['missing_values'].values())}",
          f"- Unique drugs: {a['unique_drugs']}; unique excipients: {a['unique_excipients']}; "
          f"unique molecules: {a['molecules_total']} (in both roles: {a['molecules_in_both_roles']})",
          f"- RDKit-invalid SMILES: {len(a['rdkit_invalid_smiles'])}; multi-component SMILES: "
          f"{a['multi_component_smiles(salts/mixtures)']}",
          f"- Pair multiplicity (copies in raw file -> #pairs): {a['pair_multiplicity_counts']}", ""]
    for r in ("drug", "excipient"):
        c = a[f"{r}_label_concentration"]
        L.append(f"- {r}s: {c['entities']} total, {c['always_incompatible']} always-incompatible, "
                 f"{c['always_compatible']} always-compatible, {c['mixed']} mixed")
    L += ["", "## Discrepancies vs. paper (paper value, ours)", ""]
    for k, v in a["discrepancies_vs_paper"].items():
        L.append(f"- {k}: {v}")
    return "\n".join(L) + "\n"
