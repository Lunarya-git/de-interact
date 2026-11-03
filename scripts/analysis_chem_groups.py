#!/usr/bin/env python
"""Analysis-only (no training, no change to existing pipeline): chemistry-only structure groups vs fingerprint collisions,
and feasibility of the planned tuning/evaluation entity pools.  Reads data/processed/{structure_groups,pairs,molecules}.csv
(columns smiles/flat/ikey already computed by 03_make_splits.py) and writes data/processed/chem_group_preview.json.
chem_group = union of identical SMILES | identical stereo-stripped canonical SMILES | identical InChIKey connectivity layer.
Identical PubChem fingerprint is deliberately NOT a grouping key (collisions are reported separately)."""
import json, pathlib, numpy as np, pandas as pd
root = pathlib.Path(__file__).resolve().parents[1]; P = root / "data" / "processed"
m = pd.read_csv(P / "structure_groups.csv"); pairs = pd.read_csv(P / "pairs.csv"); mol = pd.read_csv(P / "molecules.csv")
n = len(m); par = list(range(n))
def find(i):
    while par[i] != i: par[i] = par[par[i]]; i = par[i]
    return i
for col in ("smiles", "flat", "ikey"):
    first = {}
    for i, k in enumerate(m[col]):
        if pd.isna(k): continue
        if k in first: par[find(i)] = find(first[k])
        else: first[k] = i
m["chem_group"] = m.groupby(pd.Series([find(i) for i in range(n)]))["cid"].transform("min").astype(int)
sz = m.groupby("chem_group").size(); g = dict(zip(m.cid, m.chem_group))
t = pairs.assign(a=pairs.drug_cid.map(g), b=pairs.excipient_cid.map(g)); t["u"] = [tuple(sorted(x)) for x in zip(t.a, t.b)]
gp = t.groupby("u")["label"].agg(["nunique", "size", "max"])
rep = {"n_molecules": n, "chem_group": {"n_groups": int(len(sz)), "n_multi_member_groups": int((sz > 1).sum()),
       "n_molecules_in_multi_member_groups": int(sz[sz > 1].sum()), "max_group_size": int(sz.max()),
       "n_unique_group_pairs": int(len(gp)), "cid_pairs_collapsed": int(len(pairs) - len(gp)),
       "group_pairs_incompatible": int((gp["max"] == 1).sum()), "group_pairs_with_conflicting_labels": int((gp["nunique"] > 1).sum()),
       "cid_pairs_in_conflicting_group_pairs": int(gp.loc[gp["nunique"] > 1, "size"].sum())}}
# fingerprint collisions among DISTINCT chemical groups (representative molecule per group)
r = m.drop_duplicates("chem_group"); c = r.groupby("fp_hash")["chem_group"].nunique()
rep["fp_collisions"] = {"distinct_fingerprints": int(r.fp_hash.nunique()), "fingerprints_shared_by_distinct_chem_groups": int((c > 1).sum()),
                        "chem_groups_involved": int(c[c > 1].sum()), "max_groups_per_fingerprint": int(c.max())}
fp = dict(zip(m.cid, m.fp_hash)); t["fpk"] = [tuple(sorted((fp[a], fp[b]))) for a, b in zip(t.drug_cid, t.excipient_cid)]
q = t.groupby("fpk").agg(n_chem_pairs=("u", "nunique"), lab_min=("label", "min"), lab_max=("label", "max"))
rep["fp_collisions"]["fingerprint_pair_keys"] = int(len(q))
rep["fp_collisions"]["fp_pair_keys_covering_several_chem_pairs"] = int((q.n_chem_pairs > 1).sum())
rep["fp_collisions"]["fp_pair_keys_with_conflicting_labels_(identical input to FP models)"] = int((q.lab_min != q.lab_max).sum())
# planned entity pools: E = evaluation-only entity pool (label-blind, per role, dual-role groups always tuning pool D)
role = mol.set_index("cid").roles.to_dict(); gr = m.assign(role=m.cid.map(role)).groupby("chem_group")["role"].agg(lambda s: "both" if s.nunique() > 1 or (s == "both").any() else s.iloc[0])
dr = sorted(gr[gr == "drug"].index); ex = sorted(gr[gr == "excipient"].index)
t["drole"] = t.drug_cid.map(g); t["erole"] = t.excipient_cid.map(g); tt = t.drop_duplicates("u")
out = []
for seed in range(20):
    rs = np.random.RandomState(seed); E = set(rs.choice(dr, int(round(.25 * len(dr))), replace=False)) | set(rs.choice(ex, int(round(.25 * len(ex))), replace=False))
    d_in = tt.drole.isin(E); e_in = tt.erole.isin(E)
    out.append({"seed": seed, "pairs_both_in_E (cold_both pool)": int((d_in & e_in).sum()), "pos_both_in_E": int(tt.label[d_in & e_in].sum()),
                "pairs_drug_in_E": int(d_in.sum()), "pos_drug_in_E": int(tt.label[d_in].sum()),
                "pairs_exc_in_E": int(e_in.sum()), "pos_exc_in_E": int(tt.label[e_in].sum()),
                "pairs_both_in_D (tuning pool)": int((~d_in & ~e_in).sum()), "pos_both_in_D": int(tt.label[~d_in & ~e_in].sum())})
o = pd.DataFrame(out); rep["entity_pool_preview_E25pct"] = {"mean": o.mean().round(1).to_dict(), "min": o.min().to_dict()}
(P / "chem_group_preview.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
