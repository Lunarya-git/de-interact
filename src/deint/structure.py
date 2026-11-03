"""Structure-level identity groups (guards against chemical-structure leakage in cold-start splits).

PubChem often holds several CIDs for what is, for a fingerprint model, the same molecule: stereo-isomers,
anomers (lactose has 8 CIDs here), hydrate/salt records, tautomer or charge-state records.  If one CID is in
training and another in the test set, a "cold-start" split is not cold.  We therefore group molecules whose
structure is identical at one of these levels and hold groups out together:

  struct_group  union of molecules sharing ANY of: identical SMILES string | identical stereo-stripped
                canonical SMILES | identical InChIKey connectivity layer | identical PubChem fingerprint
                (identical model input => indistinguishable to the baseline ANN).  DEFAULT leakage unit.
  parent_group  struct_group additionally merged by identical largest-fragment (salt/hydrate stripped)
                stereo-stripped SMILES.  Stricter; used as a sensitivity analysis.

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
    Returns one row per molecule with struct_group / parent_group ids (= smallest CID in the group)."""
    row = {int(c): i for i, c in enumerate(cids)}
    mol = molecules[["cid", "smiles"]].copy().reset_index(drop=True)
    keys = [structure_keys(s) for s in mol.smiles]
    mol["flat"] = [k["flat"] for k in keys]; mol["ikey"] = [k["ikey"] for k in keys]; mol["parent"] = [k["parent"] for k in keys]
    mol["fp_hash"] = [np.packbits(fp[row[int(c)]].astype(np.uint8)).tobytes().hex() for c in mol.cid]
    uf = _UF(len(mol))
    for col in ("smiles", "flat", "ikey", "fp_hash"): _merge(uf, mol[col].tolist())
    def label(uf):
        root = [uf.find(i) for i in range(len(mol))]
        return mol.groupby(pd.Series(root))["cid"].transform("min").astype(int).values
    mol["struct_group"] = label(uf)
    _merge(uf, mol["parent"].tolist())
    mol["parent_group"] = label(uf)
    for g in ("struct_group", "parent_group"):
        mol[f"n_in_{g}"] = mol.groupby(g)["cid"].transform("size")
    return mol


def structure_report(mol: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Numbers quoted in the design document (labels used for REPORTING only)."""
    rep = {"n_molecules": int(len(mol))}
    for g in ("struct_group", "parent_group"):
        sizes = mol.groupby(g).size()
        m = dict(zip(mol.cid, mol[g]))
        t = pairs.assign(dg=pairs.drug_cid.map(m), eg=pairs.excipient_cid.map(m))
        t["u"] = [tuple(sorted(x)) for x in zip(t.dg, t.eg)]
        gg = t.groupby("u")["label"].agg(["nunique", "size", "max"])
        rep[g] = {"n_groups": int(len(sizes)), "n_multi_member_groups": int((sizes > 1).sum()),
                  "n_molecules_in_multi_member_groups": int(sizes[sizes > 1].sum()), "max_group_size": int(sizes.max()),
                  "n_unique_group_pairs": int(len(gg)), "cid_pairs_collapsed": int(len(pairs) - len(gg)),
                  "group_pairs_with_conflicting_labels": int((gg["nunique"] > 1).sum()),
                  "cid_pairs_in_conflicting_group_pairs": int(gg.loc[gg["nunique"] > 1, "size"].sum()),
                  "group_pairs_incompatible": int((gg["max"] == 1).sum())}
    return rep
