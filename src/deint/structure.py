"""Structure-level identity groups (guards against chemical-structure leakage in cold-start splits).

PubChem often holds several CIDs for what is, for a fingerprint model, the same molecule: stereo-isomers,
anomers (lactose has 8 CIDs here), hydrate/salt records, tautomer or charge-state records.  If one CID is in
training and another in the test set, a "cold-start" split is not cold.  We therefore group molecules whose
structure is identical at one of these levels and hold groups out together:

  chem_group    union of molecules sharing ANY of: identical SMILES string | identical stereo-stripped
                canonical SMILES | identical InChIKey connectivity layer.
                ** Identical PubChem fingerprint is deliberately NOT a grouping key here **:
                different molecules can collide to the same fingerprint and must remain
                separate chemical entities.  This is the PRIMARY (corrected) leakage unit.

  struct_group  (legacy) = chem_group PLUS merging by identical PubChem fingerprint.  Kept for the
                sensitivity analysis (E5) and backward compatibility.  Do NOT use as the default
                for evaluation — it incorrectly merges chemically distinct molecules.

  parent_group  chem_group additionally merged by identical largest-fragment (salt/hydrate stripped)
                stereo-stripped SMILES.  Stricter sensitivity variant.

Fingerprint collisions (molecules with distinct chem_groups but identical fingerprints) are
analysed separately in E5; they are never merged in chem_group.

Pure RDKit + numpy; no labels are used (grouping cannot depend on the target).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _parse(smiles: str):
    """RDKit parse with a fall-back for hypervalent species that fail sanitisation (e.g. ClF3)."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is not None:
            m.UpdatePropertyCache(strict=False)
    return m


def structure_keys(smiles: str) -> dict:
    from rdkit import Chem
    m = _parse(smiles)
    if m is None:
        return {"flat": None, "ikey": None, "parent": None}
    out = {}
    flat = Chem.Mol(m); Chem.RemoveStereochemistry(flat)
    try: out["flat"] = Chem.MolToSmiles(flat)
    except Exception: out["flat"] = None
    try:
        from rdkit.Chem import inchi
        out["ikey"] = inchi.MolToInchiKey(m).split("-")[0] or None
    except Exception:
        out["ikey"] = None
    try:
        frags = Chem.GetMolFrags(flat, asMols=True, sanitizeFrags=False)
        big = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        out["parent"] = Chem.MolToSmiles(big)
    except Exception:
        out["parent"] = None
    return out


class _UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i
    def union(self, i, j): self.p[self.find(j)] = self.find(i)


def _merge(uf: _UF, keys) -> None:
    first = {}
    for i, k in enumerate(keys):
        if k is None or (isinstance(k, float) and np.isnan(k)): continue
        if k in first: uf.union(first[k], i)
        else: first[k] = i


def build_structure_groups(molecules: pd.DataFrame, fp: np.ndarray, cids: np.ndarray) -> pd.DataFrame:
    """molecules: table with cid, smiles.  fp/cids: fingerprint matrix and its CID order.

    Builds three group columns:
      chem_group   (PRIMARY): chemistry-only; merges identical SMILES, stereo-stripped SMILES,
                              InChIKey connectivity.  Fingerprint hash is NOT a key here so that
                              fingerprint-colliding but chemically distinct molecules remain separate.
      struct_group (LEGACY):  chem_group + fingerprint hash merge (kept for E5 sensitivity analysis).
      parent_group (STRICT):  chem_group + parent-fragment merge (stricter cold-start sensitivity).

    All group ids are the minimum CID in the group.
    Returns one row per molecule with all three group columns plus n_in_* size columns.
    """
    row = {int(c): i for i, c in enumerate(cids)}
    mol = molecules[["cid", "smiles"]].copy().reset_index(drop=True)
    keys = [structure_keys(s) for s in mol.smiles]
    mol["flat"]    = [k["flat"]   for k in keys]
    mol["ikey"]    = [k["ikey"]   for k in keys]
    mol["parent"]  = [k["parent"] for k in keys]
    mol["fp_hash"] = [np.packbits(fp[row[int(c)]].astype(np.uint8)).tobytes().hex() for c in mol.cid]

    def label(uf_obj):
        root = [uf_obj.find(i) for i in range(len(mol))]
        return mol.groupby(pd.Series(root))["cid"].transform("min").astype(int).values

    # ── chem_group: chemistry only (NO fp_hash) ──────────────────────────────
    uf_chem = _UF(len(mol))
    for col in ("smiles", "flat", "ikey"):
        _merge(uf_chem, mol[col].tolist())
    mol["chem_group"] = label(uf_chem)

    # ── struct_group: chem + fp_hash (legacy; kept for backward compat) ───────
    import copy as _copy
    uf_struct = _copy.deepcopy(uf_chem)
    _merge(uf_struct, mol["fp_hash"].tolist())
    mol["struct_group"] = label(uf_struct)

    # ── parent_group: chem + parent-fragment (stricter) ───────────────────────
    uf_parent = _copy.deepcopy(uf_chem)
    _merge(uf_parent, mol["parent"].tolist())
    mol["parent_group"] = label(uf_parent)

    for g in ("chem_group", "struct_group", "parent_group"):
        mol[f"n_in_{g}"] = mol.groupby(g)["cid"].transform("size")
    return mol


def structure_report(mol: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Numbers quoted in the design document (labels used for REPORTING only).

    Reports stats for all three group types: chem_group (primary), struct_group (legacy),
    parent_group (strict).  Also reports fingerprint collision stats separately.
    """
    rep = {"n_molecules": int(len(mol))}
    groups = [g for g in ("chem_group", "struct_group", "parent_group") if g in mol.columns]
    for g in groups:
        sizes = mol.groupby(g).size()
        gmap  = dict(zip(mol.cid, mol[g]))
        t = pairs.assign(dg=pairs.drug_cid.map(gmap), eg=pairs.excipient_cid.map(gmap))
        t["u"] = [tuple(sorted(x)) for x in zip(t.dg, t.eg)]
        gg = t.groupby("u")["label"].agg(["nunique", "size", "max"])
        rep[g] = {
            "n_groups":                         int(len(sizes)),
            "n_multi_member_groups":            int((sizes > 1).sum()),
            "n_molecules_in_multi_member_groups": int(sizes[sizes > 1].sum()),
            "max_group_size":                   int(sizes.max()),
            "n_unique_group_pairs":             int(len(gg)),
            "cid_pairs_collapsed":              int(len(pairs) - len(gg)),
            "group_pairs_with_conflicting_labels": int((gg["nunique"] > 1).sum()),
            "cid_pairs_in_conflicting_group_pairs": int(gg.loc[gg["nunique"] > 1, "size"].sum()),
            "group_pairs_incompatible":         int((gg["max"] == 1).sum()),
        }
    # Fingerprint collision report (separate from grouping)
    if "fp_hash" in mol.columns and "chem_group" in mol.columns:
        rep_chem = mol.drop_duplicates("chem_group")
        c = rep_chem.groupby("fp_hash")["chem_group"].nunique()
        rep["fp_collisions"] = {
            "distinct_fingerprints":                       int(rep_chem.fp_hash.nunique()),
            "fingerprints_shared_by_distinct_chem_groups": int((c > 1).sum()),
            "chem_groups_involved":                        int(c[c > 1].sum()),
            "max_chem_groups_per_fingerprint":             int(c.max()),
        }
    return rep
