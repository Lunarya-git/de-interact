"""Data splits with explicit leakage assertions.

Split kinds (all operate on the de-duplicated pair table unless noted):
  paper_rows     ROW-level stratified split of the supplied raw CSV (duplicates NOT removed): the
                 paper-faithful (leaky) protocol.  Operates on raw_labeled.csv.
  random_pair    stratified random split of unique pairs (duplicate-leak free, entities shared).
  cold_drug      whole drugs held out: no val/test drug is in training (any role).
  cold_excipient whole excipients held out.
  cold_both      each molecule is assigned to ONE of train/val/test; a pair is kept only if both
                 members belong to the same partition (pairs straddling partitions are discarded).
Returned object: {"train": [...], "val": [...], "test": [...]} of integer row positions + report.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

KINDS = ("paper_rows", "random_pair", "cold_drug", "cold_excipient", "cold_both")


def _grouped_table(table: pd.DataFrame, groups: dict | None) -> pd.DataFrame:
    """Replace molecule CIDs by structure-group ids (role-agnostic id space) so that every downstream rule
    (cold entity sets, pair identity) is evaluated on STRUCTURES, not on database records."""
    if groups is None:
        return table
    t = table.copy()
    t["drug_cid"] = t["drug_cid"].map(groups); t["excipient_cid"] = t["excipient_cid"].map(groups)
    if t[["drug_cid", "excipient_cid"]].isna().any().any():
        raise KeyError("a molecule has no structure group - rebuild data/processed/structure_groups.csv")
    return t


def _pair_key_split(table: pd.DataFrame, fr: dict, seed: int):
    """Stratified split on UNORDERED structure-pair keys: rows sharing a key never straddle partitions."""
    key = np.array([f"{min(a, b)}|{max(a, b)}" for a, b in zip(table.drug_cid.values, table.excipient_cid.values)])
    ukeys, inv = np.unique(key, return_inverse=True)
    yk = np.zeros(len(ukeys), dtype=int); np.maximum.at(yk, inv, table["label"].values.astype(int))
    tr, va, te = _stratified(yk, fr, seed)
    part = np.empty(len(ukeys), dtype=int); part[tr], part[va], part[te] = 0, 1, 2
    p = part[inv]
    return tuple(np.where(p == k)[0] for k in (0, 1, 2))


def _stratified(y: np.ndarray, fr: dict, seed: int):
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fr["train"], stratify=y, random_state=seed)
    rel = fr["val"] / (fr["val"] + fr["test"])
    va, te = train_test_split(rest, train_size=rel, stratify=y[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def _valid(y, tr, va, te, min_pos):
    if min(len(tr), len(va), len(te)) == 0:
        return False
    if y[tr].min() == y[tr].max():
        return False
    return y[va].sum() >= min_pos and y[te].sum() >= min_pos and (1 - y[va]).sum() > 0 and (1 - y[te]).sum() > 0


def _group_split(pairs: pd.DataFrame, col: str, other: str, seed: int, cfg: dict):
    """Hold out whole groups of `col` (drug or excipient).  `other` is the other role's column; the
    molecule sets of both roles are compared by CID so a molecule in both roles cannot leak."""
    fr, y = cfg["split"], pairs["label"].values
    sizes_all = pairs.groupby(col).size()
    for attempt in range(fr["max_attempts"]):
        rng = np.random.default_rng(seed * 100_003 + attempt)
        groups = sizes_all.index.values.copy(); rng.shuffle(groups)
        cum = np.cumsum(sizes_all.reindex(groups).values) / len(pairs)
        part = np.where(cum <= fr["train"], 0, np.where(cum <= fr["train"] + fr["val"], 1, 2))
        gmap = dict(zip(groups, part))
        p = pairs[col].map(gmap).values
        held_val = set(groups[part == 1]); held_test = set(groups[part == 2])
        a, b = pairs["drug_cid"].values, pairs["excipient_cid"].values
        # molecule-level exclusion (role-agnostic)
        def touches(S): return np.isin(a, list(S)) | np.isin(b, list(S))
        tr = np.where((p == 0) & ~touches(held_val) & ~touches(held_test))[0]
        va = np.where((p == 1) & ~touches(held_test))[0]
        te = np.where((p == 2) & ~touches(held_val))[0]
        if _valid(y, tr, va, te, fr["min_pos_eval"]):
            return tr, va, te, attempt
    raise RuntimeError(f"No valid {col} split found for seed {seed}; relax min_pos_eval")


def _cold_both(pairs: pd.DataFrame, seed: int, cfg: dict):
    fr, y = cfg["split"], pairs["label"].values
    probs = np.array(cfg["split"]["cold_both_probs"], dtype=float)
    mols = np.unique(np.concatenate([pairs["drug_cid"].values, pairs["excipient_cid"].values]))
    a, b = pairs["drug_cid"].values, pairs["excipient_cid"].values
    for attempt in range(fr["max_attempts"]):
        rng = np.random.default_rng(seed * 100_003 + attempt)
        assign = dict(zip(mols, rng.choice(3, size=len(mols), p=probs)))
        pa = np.array([assign[x] for x in a]); pb = np.array([assign[x] for x in b])
        keep = pa == pb
        tr, va, te = (np.where(keep & (pa == k))[0] for k in (0, 1, 2))
        if _valid(y, tr, va, te, fr["min_pos_eval_cold_both"]):
            return tr, va, te, attempt
    raise RuntimeError(f"No valid cold_both split for seed {seed}")


def make_split(table: pd.DataFrame, kind: str, seed: int, cfg: dict, groups: dict | None = None) -> dict:
    """`table` must have columns drug_cid, excipient_cid, label (pairs, or raw rows for paper_rows).
    groups: optional {cid: structure-group id}.  When given, every held-out / identity rule is applied to
    structure groups (paper_rows ignores it: it reproduces the paper's row-level protocol on purpose)."""
    if kind not in KINDS:
        raise ValueError(f"unknown split kind {kind}")
    if kind == "paper_rows":
        groups = None
    table = _grouped_table(table, groups)
    y = table["label"].values
    attempt = 0
    if kind == "paper_rows" or (kind == "random_pair" and groups is None):
        tr, va, te = _stratified(y, cfg["split"], seed)
    elif kind == "random_pair":
        tr, va, te = _pair_key_split(table, cfg["split"], seed)
        if not _valid(y, tr, va, te, cfg["split"]["min_pos_eval"]):
            raise RuntimeError(f"random_pair (grouped) split invalid for seed {seed}")
    elif kind == "cold_drug":
        tr, va, te, attempt = _group_split(table, "drug_cid", "excipient_cid", seed, cfg)
    elif kind == "cold_excipient":
        tr, va, te, attempt = _group_split(table, "excipient_cid", "drug_cid", seed, cfg)
    else:
        tr, va, te, attempt = _cold_both(table, seed, cfg)
    split = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
    split["report"] = leakage_report(table, split, kind)
    split["report"]["attempt"] = attempt
    split["report"]["leakage_unit"] = "structure_group" if groups is not None else "cid"
    assert_no_leakage(table, split, kind, unordered_pairs=groups is not None)
    return split


def leakage_report(table: pd.DataFrame, split: dict, kind: str) -> dict:
    d, e, y = table["drug_cid"].values, table["excipient_cid"].values, table["label"].values
    tr = np.array(split["train"]); rep = {"kind": kind}
    trpairs = set(zip(d[tr], e[tr])); trd, tre = set(d[tr]), set(e[tr])
    trmol = trd | tre
    for name in ("train", "val", "test"):
        ix = np.array(split[name])
        rep[name] = {"n": int(len(ix)), "n_pos": int(y[ix].sum()),
                     "pos_rate": float(y[ix].mean()) if len(ix) else None}
        if name != "train":
            rep[name].update({
                "frac_pair_seen_in_train": float(np.mean([(a, b) in trpairs for a, b in zip(d[ix], e[ix])])),
                "frac_drug_seen_in_train": float(np.mean([a in trd for a in d[ix]])),
                "frac_excipient_seen_in_train": float(np.mean([b in tre for b in e[ix]])),
                "frac_any_molecule_seen_in_train": float(np.mean([(a in trmol) or (b in trmol) for a, b in zip(d[ix], e[ix])]))})
    return rep


def assert_no_leakage(table: pd.DataFrame, split: dict, kind: str, unordered_pairs: bool = False) -> None:
    tr, va, te = (set(split[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "row index overlap between partitions"
    d, e = table["drug_cid"].values, table["excipient_cid"].values
    if kind == "paper_rows":
        return                                   # duplicates are expected here by design
    pair = (lambda S: {(min(d[i], e[i]), max(d[i], e[i])) for i in S}) if unordered_pairs else (lambda S: {(d[i], e[i]) for i in S})
    assert not (pair(tr) & pair(va)) and not (pair(tr) & pair(te)) and not (pair(va) & pair(te)), \
        "identical (drug, excipient) pair in two partitions"
    mol = lambda S: {d[i] for i in S} | {e[i] for i in S}
    drug = lambda S: {d[i] for i in S}; exc = lambda S: {e[i] for i in S}
    if kind in ("cold_drug", "cold_excipient"):
        ent = drug if kind == "cold_drug" else exc
        Hv, Ht = ent(va), ent(te)                  # held-out entities (in the held-out ROLE)
        assert not (Hv & Ht), "same held-out entity in val and test"
        assert not (mol(tr) & (Hv | Ht)), "held-out entity appears in training (any role)"
        assert not (mol(te) & Hv) and not (mol(va) & Ht), "held-out entity leaks between val and test"
    elif kind == "cold_both":
        assert not (mol(tr) & mol(va)) and not (mol(tr) & mol(te)) and not (mol(va) & mol(te)), \
            "a molecule appears in two partitions"
