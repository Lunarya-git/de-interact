"""Comprehensive tests for DE-INTERACT + DE-CrossNet pipeline.

All tests run on synthetic data; no real dataset or Java (PaDEL) needed.
Tests are grouped:
  - Data cleaning
  - Splits and leakage assertions (including chem grouping)
  - Structure groups (chem_group vs struct_group)
  - Entity pool independence
  - Fingerprint models (ConcatMLP, SiameseNet)
  - Graph construction (graphs.py)
  - GNN models (gnn.py): symmetry, forward pass, masking
  - Attention masking and padding invariance
  - Training, checkpoints, resuming
  - End-to-end smoke tests (FP models)
  - GNN smoke test (CPU)
"""
import copy, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from deint import data, metrics, splits
from deint.baselines import EntityPrior
from deint.models import ConcatMLP, SiameseNet, n_params
from deint.train import load_checkpoint, save_checkpoint
from deint.utils import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["split"].update(min_pos_eval=3, min_pos_eval_cold_both=1, max_attempts=2000, leakage_unit="cid")
    return c


def synth_pairs(n_drug=40, n_exc=30, n_pairs=600, seed=0):
    rng = np.random.default_rng(seed)
    drugs = np.arange(1000, 1000 + n_drug); excs = np.arange(5000, 5000 + n_exc)
    excs[:3] = drugs[:3]                                   # 3 molecules used in BOTH roles
    seen, rows = set(), []
    while len(rows) < n_pairs:
        d, e = int(rng.choice(drugs)), int(rng.choice(excs))
        if d != e and (d, e) not in seen:
            seen.add((d, e)); rows.append((d, e, int(rng.random() < 0.2), 2))
    return pd.DataFrame(rows, columns=["drug_cid", "excipient_cid", "label", "n_raw_rows"])


# ════════════════════════════════════════════════════════════════════════════
# Data cleaning
# ════════════════════════════════════════════════════════════════════════════

def test_clean_dedup_and_label_map():
    raw = pd.DataFrame({"drug_cid": [1, 1, 2], "excipient_cid": [9, 9, 9],
                         "drug_smiles": ["CC", "CC", "CCO"],
                         "excipient_smiles": ["O", "O", "O"],
                         "output_value": ["Imcompatible"] * 2 + ["Compatible"]})
    pairs, mols, lab = data.clean(raw)
    assert len(pairs) == 2 and pairs.loc[pairs.drug_cid == 1, "label"].item() == 1
    assert pairs.loc[pairs.drug_cid == 1, "n_raw_rows"].item() == 2
    assert len(mols) == 3 and mols.rdkit_valid.all()


def test_conflicting_labels_raise():
    raw = pd.DataFrame({"drug_cid": [1, 1], "excipient_cid": [9, 9],
                         "drug_smiles": ["CC"] * 2, "excipient_smiles": ["O"] * 2,
                         "output_value": ["Compatible", "Imcompatible"]})
    with pytest.raises(ValueError):
        data.clean(raw)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        data.add_label(pd.DataFrame({"output_value": ["maybe"]}))


# ════════════════════════════════════════════════════════════════════════════
# Splits and leakage
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("kind", ["random_pair", "cold_drug", "cold_excipient", "cold_both"])
def test_splits_leakage_free(cfg, kind):
    t = synth_pairs()
    for seed in (0, 1):
        sp = splits.make_split(t, kind, seed, cfg)
        r = sp["report"]
        assert r["test"]["frac_pair_seen_in_train"] == 0
        if kind == "cold_drug":
            assert r["test"]["frac_drug_seen_in_train"] == 0
        if kind == "cold_excipient":
            assert r["test"]["frac_excipient_seen_in_train"] == 0
        if kind == "cold_both":
            assert r["test"]["frac_any_molecule_seen_in_train"] == 0


def test_leakage_detector_catches_injected_leak(cfg):
    t = synth_pairs(); sp = splits.make_split(t, "cold_drug", 0, cfg)
    leaky = copy.deepcopy(sp); leaky["train"].append(leaky["test"][0])
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(t, leaky, "cold_drug")
    d = t.drug_cid.values; te_drug = d[sp["test"][0]]
    extra = [i for i in range(len(t)) if d[i] == te_drug and i not in sp["test"]]
    if extra:
        leaky2 = copy.deepcopy(sp); leaky2["train"].append(extra[0])
        with pytest.raises(AssertionError):
            splits.assert_no_leakage(t, leaky2, "cold_drug")


def test_paper_rows_split_is_leaky_by_design(cfg):
    base = synth_pairs(n_pairs=300); raw = pd.concat([base, base], ignore_index=True)
    sp = splits.make_split(raw, "paper_rows", 0, cfg)
    assert sp["report"]["test"]["frac_pair_seen_in_train"] > 0.4


def test_splits_deterministic(cfg):
    t = synth_pairs()
    assert splits.make_split(t, "cold_drug", 3, cfg)["test"] == splits.make_split(t, "cold_drug", 3, cfg)["test"]


# ════════════════════════════════════════════════════════════════════════════
# Structure groups: chem_group vs struct_group
# ════════════════════════════════════════════════════════════════════════════

def test_chem_group_does_NOT_merge_fp_collision():
    """Molecules with different chemistry but same fingerprint must stay in separate chem_groups."""
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2, 3], "smiles": ["CCO", "CCCO", "c1ccccc1"]})
    fp = np.zeros((3, 881), np.uint8)
    # Give CIDs 1 and 2 identical fingerprints (collision) but different SMILES
    fp[0, 0] = 1; fp[1, 0] = 1  # same fp_hash
    fp[2, 5] = 1
    g = build_structure_groups(mols, fp, mols.cid.values).set_index("cid")
    # chem_group: 1 and 2 differ (different SMILES, different ikey) → different groups
    assert g.chem_group[1] != g.chem_group[2], "chem_group must NOT merge on fp collision alone"
    # struct_group (legacy): 1 and 2 SHOULD be merged (same fp_hash)
    assert g.struct_group[1] == g.struct_group[2], "struct_group should still merge fp-identical mols"


def test_chem_group_merges_stereo_isomers():
    """L-alanine, D-alanine and flat alanine should all share a chem_group."""
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2, 3, 4],
                          "smiles": ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(N)C(=O)O", "CCO"]})
    fp = np.zeros((4, 881), np.uint8); fp[0, 1] = 1; fp[1, 2] = 1; fp[2, 3] = 1; fp[3, 9] = 1
    g = build_structure_groups(mols, fp, mols.cid.values).set_index("cid")
    assert g.chem_group[1] == g.chem_group[2] == g.chem_group[3]
    assert g.chem_group[4] != g.chem_group[1]


def test_parent_group_merges_salts():
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2], "smiles": ["CCN", "CCN.Cl"]})
    fp = np.zeros((2, 881), np.uint8); fp[0, 1] = 1; fp[1, 2] = 1
    g = build_structure_groups(mols, fp, mols.cid.values).set_index("cid")
    assert g.chem_group[1] != g.chem_group[2]    # different SMILES → separate chem_groups
    assert g.parent_group[1] == g.parent_group[2] # but same largest fragment


def test_structure_parse_fallback_for_hypervalent():
    from deint.structure import structure_keys
    k = structure_keys("FCl(F)F")   # ClF3 fails normal RDKit sanitisation
    assert k["flat"] is not None, "ClF3 fallback parse must succeed"


@pytest.mark.parametrize("kind", ["random_pair", "cold_drug", "cold_excipient", "cold_both"])
def test_grouped_splits_hold_out_whole_chem_groups(cfg, kind):
    t = synth_pairs()
    mols = np.unique(np.concatenate([t.drug_cid, t.excipient_cid]))
    groups = {int(m): int(m) for m in mols}
    for m in mols[::4]: groups[int(m)] = int(mols[0 if m % 2 else 1])  # collapse some CIDs
    for seed in (0, 1):
        try:
            sp = splits.make_split(t, kind, seed, cfg, groups)
        except RuntimeError:
            pytest.skip("no valid split on tiny synthetic data")
            return
        gt = splits._grouped_table(t, groups)
        splits.assert_no_leakage(gt, sp, kind, unordered_pairs=True)
        assert sp["report"]["leakage_unit"] == "structure_group"
        assert sp["report"]["test"]["frac_pair_seen_in_train"] == 0


def test_grouped_random_pair_keeps_equivalent_pairs_together(cfg):
    t = synth_pairs(n_pairs=300)
    groups = {int(m): int(m) // 3 for m in np.unique(np.concatenate([t.drug_cid, t.excipient_cid]))}
    sp = splits.make_split(t, "random_pair", 0, cfg, groups)
    gt = splits._grouped_table(t, groups)
    key = lambda i: tuple(sorted((gt.drug_cid.iloc[i], gt.excipient_cid.iloc[i])))
    ktr, kte = {key(i) for i in sp["train"]}, {key(i) for i in sp["test"]}
    assert not (ktr & kte)


# ════════════════════════════════════════════════════════════════════════════
# Entity pool independence
# ════════════════════════════════════════════════════════════════════════════

def test_entity_pool_independence_and_disjoint():
    from deint.entity_pool import build_entity_pools, assert_pool_independence
    mols = pd.DataFrame({
        "cid":   [1, 2, 3, 4, 5, 6, 7, 8],
        "roles": ["drug", "drug", "drug", "excipient", "excipient", "excipient", "drug+excipient", "drug"]
    })
    # Build fake structure groups (each CID is its own chem_group)
    sg = pd.DataFrame({"cid": mols.cid, "chem_group": mols.cid})
    for seed in range(5):
        pools = build_entity_pools(mols, sg, pool_frac=0.25, seed=seed)
        assert_pool_independence(pools)
        E = set(pools["E_cids"]); D = set(pools["D_cids"])
        assert not (E & D), f"E and D must be disjoint (seed {seed})"


def test_entity_pool_both_role_always_in_D():
    from deint.entity_pool import build_entity_pools
    mols = pd.DataFrame({"cid": [1, 2, 3, 4], "roles": ["drug", "excipient", "drug+excipient", "drug"]})
    sg = pd.DataFrame({"cid": mols.cid, "chem_group": mols.cid})
    pools = build_entity_pools(mols, sg, pool_frac=0.25, seed=0)
    # CID 3 (drug+excipient role) must be in D, not E
    assert 3 in pools["D_cids"], "dual-role molecules must be in pool D"
    assert 3 not in pools["E_cids"]


def test_pool_filter_pairs():
    from deint.entity_pool import filter_pairs_to_pool
    pairs = pd.DataFrame({"drug_cid": [1, 1, 2, 3], "excipient_cid": [4, 5, 5, 6], "label": [0, 1, 0, 1]})
    E_cids = [1, 4]
    filtered = filter_pairs_to_pool(pairs, E_cids)
    assert len(filtered) == 1 and filtered.iloc[0].drug_cid == 1 and filtered.iloc[0].excipient_cid == 4


# ════════════════════════════════════════════════════════════════════════════
# FP models
# ════════════════════════════════════════════════════════════════════════════

def test_concat_mlp_shapes_and_backward():
    for out_dim in (1, 2):
        m = ConcatMLP((64, 32), 0.1, out_dim); xa, xb = torch.rand(8, 881), torch.rand(8, 881)
        o = m(xa, xb); assert o.shape == ((8,) if out_dim == 1 else (8, 2))
        o.sum().backward(); assert all(p.grad is not None for p in m.parameters())
    assert n_params(ConcatMLP((1024,) * 3, 0.0, 2)) == 1762 * 1024 + 1024 + 2 * (1024 * 1024 + 1024) + 1024 * 2 + 2


@pytest.mark.parametrize("inter", ["full", "sum"])
def test_siamese_symmetric_and_backward(inter):
    m = SiameseNet((32, 16), (8,), 0.3, inter).eval()
    xa, xb = torch.rand(8, 881), torch.rand(8, 881)
    assert torch.allclose(m(xa, xb), m(xb, xa), atol=1e-6)
    m.train(); m(xa, xb).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_concat_is_not_symmetric():
    m = ConcatMLP((32,), 0.0, 1).eval()
    xa, xb = torch.rand(8, 881), torch.rand(8, 881)
    assert not torch.allclose(m(xa, xb), m(xb, xa), atol=1e-4)


def test_checkpoint_roundtrip(tmp_path):
    m = SiameseNet((16,), (8,), 0.0, "full")
    save_checkpoint(tmp_path / "c.pt", m.state_dict(), {"k": 1})
    m2 = SiameseNet((16,), (8,), 0.0, "full"); meta = load_checkpoint(tmp_path / "c.pt", m2)
    xa, xb = torch.rand(4, 881), torch.rand(4, 881)
    assert meta["k"] == 1 and torch.allclose(m.eval()(xa, xb), m2.eval()(xa, xb))


# ════════════════════════════════════════════════════════════════════════════
# Graph construction
# ════════════════════════════════════════════════════════════════════════════

def test_mol_to_graph_simple():
    from deint.graphs import mol_to_graph, ATOM_FEAT_DIM, BOND_FEAT_DIM
    g = mol_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    assert g.valid
    assert g.x.shape[1] == ATOM_FEAT_DIM
    assert g.edge_index.shape[0] == 2
    assert g.edge_attr.shape[1] == BOND_FEAT_DIM
    assert g.n_atoms == g.x.shape[0]


def test_mol_to_graph_salt_multicomponent():
    from deint.graphs import mol_to_graph
    g = mol_to_graph("CCN.Cl")
    assert g.valid
    assert g.n_atoms > 1   # keeps all atoms by default


def test_mol_to_graph_largest_fragment():
    from deint.graphs import mol_to_graph
    g_full  = mol_to_graph("CCN.Cl", use_largest_fragment=False)
    g_large = mol_to_graph("CCN.Cl", use_largest_fragment=True)
    assert g_large.n_atoms < g_full.n_atoms   # HCl stripped


def test_mol_to_graph_clf3_fallback():
    """ClF3 fails standard RDKit sanitisation; fallback must succeed."""
    from deint.graphs import mol_to_graph
    g = mol_to_graph("FCl(F)F")
    assert g.valid, "ClF3 fallback parse should produce a valid graph"
    assert g.n_atoms >= 4


def test_mol_to_graph_invalid_smiles():
    from deint.graphs import mol_to_graph
    g = mol_to_graph("THIS_IS_NOT_A_SMILES")
    assert not g.valid


def test_mol_to_graph_explicit_hydrogens():
    from deint.graphs import mol_to_graph
    g = mol_to_graph("[H]C([H])([H])[H]")  # explicit H methane
    assert g.valid and g.n_atoms >= 1


# ════════════════════════════════════════════════════════════════════════════
# Graph batching and collation
# ════════════════════════════════════════════════════════════════════════════

def test_collate_graphs_shapes():
    from deint.graphs import mol_to_graph, collate_graphs, ATOM_FEAT_DIM, BOND_FEAT_DIM
    smiles_list = ["CC", "c1ccccc1", "CCO"]
    graphs = [mol_to_graph(s) for s in smiles_list]
    x_pad, mask, ei, ea, batch, sizes = collate_graphs(graphs)
    B = len(graphs); max_n = max(g.n_atoms for g in graphs)
    assert x_pad.shape == (B, max_n, ATOM_FEAT_DIM)
    assert mask.shape == (B, max_n)
    assert mask.sum().item() == sum(g.n_atoms for g in graphs)
    assert ei.shape[0] == 2


def test_collate_graphs_padding_mask():
    from deint.graphs import mol_to_graph, collate_graphs
    g1 = mol_to_graph("CC")     # 2 atoms
    g2 = mol_to_graph("CCCCCC") # 6 atoms
    x_pad, mask, *_ = collate_graphs([g1, g2])
    assert mask[0, :2].all()  and (not mask[0, 2:].any())
    assert mask[1, :6].all()


# ════════════════════════════════════════════════════════════════════════════
# GNN models: forward pass, symmetry, attention masking
# ════════════════════════════════════════════════════════════════════════════

def _make_small_batch(smiles: str, device="cpu"):
    from deint.graphs import mol_to_graph, collate_graphs
    g = mol_to_graph(smiles)
    return collate_graphs([g], device=device)


def _rand_fp(B=1, n=881):
    return torch.rand(B, n)


@pytest.mark.parametrize("model_name", ["gnn_siamese", "de_crossnet", "de_crossnet_no_fp", "de_crossnet_self_attn", "de_crossnet_ordered_head"])
def test_gnn_forward_shape(model_name):
    from deint.gnn import build_gnn_model
    mcfg = {"d_h": 32, "n_layers": 2, "n_heads": 4, "d_fp_out": 16, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model(model_name, mcfg).eval()
    smiles_pairs = [("CC", "c1ccccc1"), ("CCO", "CCCO")]
    from deint.graphs import mol_to_graph, collate_graphs
    gA = [mol_to_graph(s[0]) for s in smiles_pairs]
    gB = [mol_to_graph(s[1]) for s in smiles_pairs]
    bA = collate_graphs(gA); bB = collate_graphs(gB)
    fpA = _rand_fp(2); fpB = _rand_fp(2)
    with torch.no_grad():
        out = model(bA, bB, fpA, fpB)
    assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"


def test_decrossnet_symmetry():
    """DE-CrossNet must produce the same logit when drug and excipient are swapped."""
    from deint.gnn import build_gnn_model
    from deint.graphs import mol_to_graph, collate_graphs
    mcfg = {"d_h": 32, "n_layers": 2, "n_heads": 4, "d_fp_out": 16, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model("de_crossnet", mcfg).eval()
    gA = collate_graphs([mol_to_graph("CC")])
    gB = collate_graphs([mol_to_graph("CCO")])
    fpA = torch.rand(1, 881); fpB = torch.rand(1, 881)
    with torch.no_grad():
        out_AB = model(gA, gB, fpA, fpB)
        out_BA = model(gB, gA, fpB, fpA)
    assert torch.allclose(out_AB, out_BA, atol=1e-5), \
        f"DE-CrossNet symmetry violation: {out_AB} vs {out_BA}"


def test_gnn_siamese_symmetry():
    from deint.gnn import build_gnn_model
    from deint.graphs import mol_to_graph, collate_graphs
    mcfg = {"d_h": 32, "n_layers": 2, "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model("gnn_siamese", mcfg).eval()
    gA = collate_graphs([mol_to_graph("CC")])
    gB = collate_graphs([mol_to_graph("CCCCO")])
    with torch.no_grad():
        assert torch.allclose(model(gA, gB), model(gB, gA), atol=1e-5)


def test_decrossnet_ordered_head_is_not_symmetric():
    from deint.gnn import build_gnn_model
    from deint.graphs import mol_to_graph, collate_graphs
    mcfg = {"d_h": 32, "n_layers": 2, "n_heads": 4, "d_fp_out": 16, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model("de_crossnet_ordered_head", mcfg).eval()
    gA = collate_graphs([mol_to_graph("CC")])
    gB = collate_graphs([mol_to_graph("CCO")])
    fpA = torch.rand(1, 881); fpB = torch.rand(1, 881)
    with torch.no_grad():
        out_AB = model(gA, gB, fpA, fpB)
        out_BA = model(gB, gA, fpB, fpA)
    assert not torch.allclose(out_AB, out_BA, atol=1e-4), "Ordered head must not be symmetric"


def test_attention_padding_invariance():
    """Padding in graph B must not affect logit for graph A when mask is correct."""
    from deint.gnn import build_gnn_model
    from deint.graphs import mol_to_graph, collate_graphs, ATOM_FEAT_DIM, BOND_FEAT_DIM
    import torch
    mcfg = {"d_h": 32, "n_layers": 2, "n_heads": 4, "d_fp_out": 16, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model("de_crossnet", mcfg).eval()
    gA = mol_to_graph("CC")       # 2 atoms
    gB_small = mol_to_graph("CCO") # 3 atoms
    gB_large  = mol_to_graph("CCCCCC")  # 6 atoms — different molecule, but different size

    bA1 = collate_graphs([gA]); bB1 = collate_graphs([gB_small])
    bA2 = collate_graphs([gA]); bB2 = collate_graphs([gB_large])
    fpA = torch.rand(1, 881)
    fpB1 = torch.rand(1, 881); fpB2 = torch.rand(1, 881)

    # The padding positions in x_pad should be all zeros and mask=False
    # Verify: if we manually set padding to random values, logit must NOT change
    x_pad2, mask2, ei2, ea2, bv2, sz2 = bB2
    x_pad2_noisy = x_pad2.clone()
    x_pad2_noisy[~mask2] = torch.rand_like(x_pad2_noisy[~mask2])  # corrupt padding
    bB2_noisy = (x_pad2_noisy, mask2, ei2, ea2, bv2, sz2)

    with torch.no_grad():
        out1 = model(bA2, bB2, fpA, fpB2)
        out2 = model(bA2, bB2_noisy, fpA, fpB2)
    assert torch.allclose(out1, out2, atol=1e-5), "Padding corruption should not affect output"


def test_gnn_backward():
    from deint.gnn import build_gnn_model
    from deint.graphs import mol_to_graph, collate_graphs
    mcfg = {"d_h": 32, "n_layers": 2, "n_heads": 4, "d_fp_out": 16, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [32]}
    model = build_gnn_model("de_crossnet", mcfg).train()
    gA = collate_graphs([mol_to_graph("CC"), mol_to_graph("CCO")])
    gB = collate_graphs([mol_to_graph("c1ccccc1"), mol_to_graph("CCCC")])
    fpA = torch.rand(2, 881, requires_grad=False)
    fpB = torch.rand(2, 881, requires_grad=False)
    out = model(gA, gB, fpA, fpB)
    out.sum().backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "All trainable parameters must receive gradients"


# ════════════════════════════════════════════════════════════════════════════
# Metrics and baselines
# ════════════════════════════════════════════════════════════════════════════

def test_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); s = np.array([.9, .8, .7, .2, .6, .1, .1, .1, .1, .1])
    m = metrics.compute_metrics(y, s, 0.5)
    assert m["confusion_matrix"] == [[5, 1], [1, 3]]
    assert m["precision_incompatible"] == pytest.approx(0.75) and m["recall_incompatible"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 0.75) / 2)
    assert m["roc_auc"] > 0.8 and 0 < m["pr_auc"] <= 1


def test_threshold_uses_given_data_only():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .4, .6, .9])
    t = metrics.best_threshold_f1(y, s)
    assert 0.4 < t <= 0.6


def test_entity_prior_unseen_gives_prior():
    m = EntityPrior(2.0).fit(np.array([1, 1, 2]), np.array([7, 8, 7]), np.array([1, 1, 0]))
    assert m.predict_score(np.array([99]), np.array([98]))[0] == pytest.approx(2 / 3, abs=1e-6)
    assert m.predict_score(np.array([1]), np.array([99]))[0] > 2 / 3


# ════════════════════════════════════════════════════════════════════════════
# Fingerprints (PaDEL; skipped if Java missing)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(shutil.which("java") is None, reason="Java not installed")
def test_padel_pubchem_fingerprint():
    from deint import features
    smi = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"]
    fps = features.padel_fingerprints(smi)
    assert all(f is not None and f.shape == (881,) and set(np.unique(f)) <= {0, 1} for f in fps)
    assert not np.array_equal(fps[0], fps[1])
    assert features.sanity_checks({smi[0]: fps[0]})["aspirin_H_bits_ok"]
    assert np.array_equal(fps[0], features.padel_fingerprints([smi[0]])[0])


# ════════════════════════════════════════════════════════════════════════════
# End-to-end smoke test: fingerprint models
# ════════════════════════════════════════════════════════════════════════════

def test_end_to_end_smoke(tmp_path, cfg):
    from deint.experiments import Context, run_experiment
    c = copy.deepcopy(cfg)
    proc = tmp_path / "proc"; proc.mkdir()
    pairs = synth_pairs(); raw = pd.concat([pairs, pairs], ignore_index=True)[["drug_cid", "excipient_cid", "label"]]
    pairs.to_csv(proc / "pairs.csv", index=False); raw.to_csv(proc / "raw_labeled.csv", index=False)
    mols = np.unique(np.concatenate([pairs.drug_cid, pairs.excipient_cid]))
    rng = np.random.default_rng(0)
    np.savez(proc / "fingerprints.npz", fp=(rng.random((len(mols), 881)) < 0.2).astype(np.uint8),
             cid=mols, ok=np.ones(len(mols), bool))
    c["paths"].update(processed_dir=str(proc), results_dir=str(tmp_path / "res"))
    c["paper_protocol"].update(epochs=3, hidden=[32, 32]); c["unified_protocol"].update(epochs=3, patience=2)
    c["models"]["concat_mlp"]["hidden"] = [32, 32]
    c["models"]["siamese"].update(encoder=[32, 16], head=[8]); c["models"]["siamese_sum"].update(encoder=[32, 16], head=[8])
    c["models"]["rf"]["n_estimators"] = 10; c["models"]["concat_mlp_reg"]["hidden"] = [32, 32]
    c["seeds"] = [0, 1]
    c["experiments"]["E1_reproduction"]["seeds"] = 1; c["experiments"]["E2_leakage_audit"]["seeds"] = 1
    c["experiments"]["E3_generalization"]["seeds"] = 2; c["experiments"]["E4_ablation"]["seeds"] = 1
    # Restrict E3/E4 to FP models only (graph models need molecules.csv)
    c["experiments"]["E3_generalization"]["models"] = ["concat_mlp", "siamese", "rf", "entity_prior"]
    c["experiments"]["E4_ablation"]["models"] = ["siamese_sum", "concat_mlp_reg"]
    c["plots"]["dpi"] = 60
    c["split"]["leakage_unit"] = "cid"
    ctx = Context(c, torch.device("cpu"))
    for e in ["E1_reproduction", "E2_leakage_audit", "E3_generalization", "E4_ablation"]:
        run_experiment(ctx, e, progress=False)
    res = tmp_path / "res"
    must = ["config.yaml", "metrics.json", "metrics.txt", "training_log.txt", "classification_report.txt",
            "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "training_curves.png",
            "predictions_test.csv"]
    d = res / "E3_generalization" / "siamese__cold_drug__seed0"
    assert all((d / f).exists() for f in must), [f for f in must if not (d / f).exists()]
    assert (d / "checkpoint_val_pr_auc.pt").exists()
    m = json.load(open(d / "metrics.json"))
    assert m["primary"]["test"]["n"] > 0 and m["split_report"]["test"]["frac_drug_seen_in_train"] == 0
    # Resume: finished runs are skipped
    mt = (d / "metrics.json").stat().st_mtime
    run_experiment(ctx, "E3_generalization", models=["siamese"], splits=["cold_drug"], progress=False)
    assert (d / "metrics.json").stat().st_mtime == mt
    # Force: old run is archived
    run_experiment(ctx, "E3_generalization", models=["siamese"], splits=["cold_drug"], seeds=[0],
                   force=True, progress=False)
    assert len(list((res / "_archive").glob("siamese__cold_drug__seed0__*"))) == 1


# ════════════════════════════════════════════════════════════════════════════
# End-to-end smoke test: GNN models (CPU)
# ════════════════════════════════════════════════════════════════════════════

def _make_gnn_test_context(tmp_path, cfg_override):
    """Build a minimal Context with molecules.csv for GNN smoke tests."""
    from deint.experiments import Context
    import copy
    c = copy.deepcopy(cfg_override)
    proc = tmp_path / "proc_gnn"; proc.mkdir(exist_ok=True)

    # Minimal molecules table with real SMILES
    smiles_list = ["CC", "CCO", "c1ccccc1", "CCCC", "CC(=O)O", "CCN",
                   "c1ccc(N)cc1", "CCCCCC", "CC(C)O", "c1cccnc1"]
    n_mol = len(smiles_list)
    mol_df = pd.DataFrame({
        "cid":         list(range(1000, 1000 + n_mol)),
        "smiles":      smiles_list,
        "roles":       (["drug"] * 5) + (["excipient"] * 5),
        "rdkit_valid": [True] * n_mol,
        "n_components": [1] * n_mol,
    })
    mol_df.to_csv(proc / "molecules.csv", index=False)

    # Generate pairs (drug=1000..1004, exc=1005..1009)
    drugs = list(range(1000, 1005)); excs = list(range(1005, 1010))
    rng = np.random.default_rng(7)
    rows = []
    for d in drugs:
        for e in excs:
            rows.append({"drug_cid": d, "excipient_cid": e, "label": int(rng.random() < 0.2), "n_raw_rows": 1})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(proc / "pairs.csv", index=False)
    raw = pd.concat([pairs, pairs], ignore_index=True)[["drug_cid", "excipient_cid", "label"]]
    raw.to_csv(proc / "raw_labeled.csv", index=False)

    all_cids = list(range(1000, 1010))
    fp = rng.integers(0, 2, (n_mol, 881)).astype(np.uint8)
    np.savez(proc / "fingerprints.npz", fp=fp, cid=np.array(all_cids),
             ok=np.ones(n_mol, bool))

    c["paths"].update(processed_dir=str(proc), results_dir=str(tmp_path / "res_gnn"))
    c["split"].update(leakage_unit="cid", min_pos_eval=1, min_pos_eval_cold_both=1)
    c["seeds"] = [0]
    return Context(c, torch.device("cpu")), c


def test_gnn_smoke_single_run(tmp_path, cfg):
    """Run a single DE-CrossNet forward+backward on tiny synthetic data."""
    from deint.gnn import build_gnn_model
    from deint.graphs import build_graph_cache
    from deint.gnn_train import fit_graph
    import pandas as pd
    smiles_list = ["CC", "CCO", "c1ccccc1", "CCCC"]
    mol_df = pd.DataFrame({"cid": [1, 2, 3, 4], "smiles": smiles_list,
                            "roles": ["drug"] * 2 + ["excipient"] * 2,
                            "rdkit_valid": [True] * 4, "n_components": [1] * 4})
    cache, _ = build_graph_cache(mol_df)

    cids_a = np.array([1, 1, 2, 2]); cids_b = np.array([3, 4, 3, 4])
    y      = np.array([1, 0, 0, 1], dtype=float)
    fp     = np.random.rand(5, 881).astype(np.float32)
    fp_map = {1: 0, 2: 1, 3: 2, 4: 3}

    def fp_by_cid(cids): return np.array([fp[fp_map[c]] for c in cids])

    mcfg = {"d_h": 16, "n_layers": 2, "n_heads": 2, "d_fp_out": 8, "n_cross": 1,
            "dropout": 0.0, "head_hidden": [16]}
    model = build_gnn_model("de_crossnet", mcfg)
    proto = {"head": "logit1", "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0,
             "batch_size": 4, "epochs": 2, "pos_weight": "auto",
             "mixed_precision": False, "max_grad_norm": 1.0}

    cid_to_row = {1: 0, 2: 1, 3: 2, 4: 3}
    out = fit_graph(
        model=model,
        graph_cache=cache,
        train_cids_a=cids_a, train_cids_b=cids_b, train_y=y,
        val_cids_a=cids_a, val_cids_b=cids_b, val_y=y,
        fp_matrix=fp_matrix,
        proto=proto, criteria=["val_pr_auc"],
        device=torch.device("cpu"), seed=0, logger=None, progress=False, patience=None,
        cid_to_row=cid_to_row,
    )
    assert out["epochs_run"] == 2
    assert "best_state" in out and "val_pr_auc" in out["best_state"]


# ════════════════════════════════════════════════════════════════════════════
# Aggregation (script 05)
# ════════════════════════════════════════════════════════════════════════════

def test_aggregation_script(tmp_path, cfg):
    from deint.experiments import Context, run_experiment
    import copy
    c = copy.deepcopy(cfg)
    proc = tmp_path / "proc_agg"; proc.mkdir()
    pairs = synth_pairs()
    raw = pd.concat([pairs, pairs], ignore_index=True)[["drug_cid", "excipient_cid", "label"]]
    pairs.to_csv(proc / "pairs.csv", index=False); raw.to_csv(proc / "raw_labeled.csv", index=False)
    mols = np.unique(np.concatenate([pairs.drug_cid, pairs.excipient_cid]))
    rng = np.random.default_rng(5)
    np.savez(proc / "fingerprints.npz", fp=(rng.random((len(mols), 881)) < 0.2).astype(np.uint8),
             cid=mols, ok=np.ones(len(mols), bool))
    c["paths"].update(processed_dir=str(proc), results_dir=str(tmp_path / "res_agg"))
    c["unified_protocol"].update(epochs=2, patience=2); c["seeds"] = [0]
    c["experiments"]["E3_generalization"].update(seeds=1, models=["concat_mlp", "entity_prior"],
                                                  splits=["random_pair", "cold_drug"])
    c["models"]["concat_mlp"]["hidden"] = [16]
    c["split"]["leakage_unit"] = "cid"
    c["plots"]["dpi"] = 60
    ctx = Context(c, torch.device("cpu"))
    run_experiment(ctx, "E3_generalization", progress=False)
    cfg_path = tmp_path / "cfg.yaml"; yaml.safe_dump(c, open(cfg_path, "w"))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "05_aggregate_results.py"),
                        "--config", str(cfg_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "res_agg" / "summary" / "summary.md").exists()
