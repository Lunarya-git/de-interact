"""Lightweight tests on small synthetic data (no full training, no real dataset needed)."""
import copy, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch, yaml

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


# ---------------- data cleaning ----------------
def test_clean_dedup_and_label_map():
    raw = pd.DataFrame({"drug_cid": [1, 1, 2], "excipient_cid": [9, 9, 9], "drug_smiles": ["CC", "CC", "CCO"],
                        "excipient_smiles": ["O", "O", "O"], "output_value": ["Imcompatible"] * 2 + ["Compatible"]})
    pairs, mols, lab = data.clean(raw)
    assert len(pairs) == 2 and pairs.loc[pairs.drug_cid == 1, "label"].item() == 1
    assert pairs.loc[pairs.drug_cid == 1, "n_raw_rows"].item() == 2
    assert len(mols) == 3 and mols.rdkit_valid.all()


def test_conflicting_labels_raise():
    raw = pd.DataFrame({"drug_cid": [1, 1], "excipient_cid": [9, 9], "drug_smiles": ["CC"] * 2,
                        "excipient_smiles": ["O"] * 2, "output_value": ["Compatible", "Imcompatible"]})
    with pytest.raises(ValueError):
        data.clean(raw)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        data.add_label(pd.DataFrame({"output_value": ["maybe"]}))


# ---------------- splits ----------------
@pytest.mark.parametrize("kind", ["random_pair", "cold_drug", "cold_excipient", "cold_both"])
def test_splits_leakage_free(cfg, kind):
    t = synth_pairs()
    for seed in (0, 1):
        sp = splits.make_split(t, kind, seed, cfg)          # asserts internally
        r = sp["report"]
        assert r["test"]["frac_pair_seen_in_train"] == 0
        if kind == "cold_drug": assert r["test"]["frac_drug_seen_in_train"] == 0
        if kind == "cold_excipient": assert r["test"]["frac_excipient_seen_in_train"] == 0
        if kind == "cold_both": assert r["test"]["frac_any_molecule_seen_in_train"] == 0


def test_leakage_detector_catches_injected_leak(cfg):
    t = synth_pairs(); sp = splits.make_split(t, "cold_drug", 0, cfg)
    leaky = copy.deepcopy(sp); leaky["train"].append(leaky["test"][0])
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(t, leaky, "cold_drug")
    # a training pair that shares the test drug (but is a different pair) must also be caught
    d = t.drug_cid.values; te_drug = d[sp["test"][0]]
    extra = [i for i in range(len(t)) if d[i] == te_drug and i not in sp["test"]]
    if extra:
        leaky2 = copy.deepcopy(sp); leaky2["train"].append(extra[0])
        with pytest.raises(AssertionError):
            splits.assert_no_leakage(t, leaky2, "cold_drug")


def test_paper_rows_split_is_leaky_by_design(cfg):
    base = synth_pairs(n_pairs=300); raw = pd.concat([base, base], ignore_index=True)    # every pair twice
    sp = splits.make_split(raw, "paper_rows", 0, cfg)
    assert sp["report"]["test"]["frac_pair_seen_in_train"] > 0.4      # measurable duplicate leakage


def test_splits_deterministic(cfg):
    t = synth_pairs()
    assert splits.make_split(t, "cold_drug", 3, cfg)["test"] == splits.make_split(t, "cold_drug", 3, cfg)["test"]


# ---------------- structure groups (chemical-structure leakage) ----------------
def test_structure_groups_merge_isomers_and_identical_fingerprints():
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2, 3, 4, 5, 6],
                         "smiles": ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(N)C(=O)O",   # L/D/flat alanine
                                    "CCO", "CCCO", "c1ccccc1.Cl"]})
    fp = np.zeros((6, 881), np.uint8); fp[3, 5] = 1; fp[4, 5] = 1; fp[5, 9] = 1   # CIDs 4,5 share a fingerprint
    g = build_structure_groups(mols, fp, mols.cid.values).set_index("cid")
    assert g.struct_group[1] == g.struct_group[2] == g.struct_group[3]
    assert g.struct_group[4] == g.struct_group[5] and g.struct_group[4] != g.struct_group[1]
    assert g.struct_group[6] not in (g.struct_group[1], g.struct_group[4])


def test_parent_group_merges_salts():
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2], "smiles": ["CCN", "CCN.Cl"]})
    fp = np.zeros((2, 881), np.uint8); fp[0, 1] = 1; fp[1, 2] = 1
    g = build_structure_groups(mols, fp, mols.cid.values).set_index("cid")
    assert g.struct_group[1] != g.struct_group[2] and g.parent_group[1] == g.parent_group[2]


def test_structure_parse_fallback_for_hypervalent():
    from deint.structure import structure_keys
    assert structure_keys("FCl(F)F")["flat"] is not None            # ClF3 fails RDKit sanitisation


@pytest.mark.parametrize("kind", ["random_pair", "cold_drug", "cold_excipient", "cold_both"])
def test_grouped_splits_hold_out_whole_structure_groups(cfg, kind):
    t = synth_pairs()
    mols = np.unique(np.concatenate([t.drug_cid, t.excipient_cid]))
    groups = {int(m): int(m) for m in mols}
    for m in mols[::4]: groups[int(m)] = int(mols[0 if m % 2 else 1])      # collapse many CIDs into 2 big groups
    for seed in (0, 1):
        try:
            sp = splits.make_split(t, kind, seed, cfg, groups)
        except RuntimeError:                                           # tiny synthetic data may admit no valid cold split
            pytest.skip("no valid split on tiny synthetic data"); return
        gt = splits._grouped_table(t, groups)
        splits.assert_no_leakage(gt, sp, kind, unordered_pairs=True)
        assert sp["report"]["leakage_unit"] == "structure_group"
        assert sp["report"]["test"]["frac_pair_seen_in_train"] == 0


def test_grouped_random_pair_keeps_equivalent_pairs_together(cfg):
    t = synth_pairs(n_pairs=300)
    groups = {int(m): int(m) // 3 for m in np.unique(np.concatenate([t.drug_cid, t.excipient_cid]))}   # 3 CIDs -> 1 group
    sp = splits.make_split(t, "random_pair", 0, cfg, groups)
    gt = splits._grouped_table(t, groups); key = lambda i: tuple(sorted((gt.drug_cid[i], gt.excipient_cid[i])))
    ktr, kte = {key(i) for i in sp["train"]}, {key(i) for i in sp["test"]}
    assert not (ktr & kte)


# ---------------- models ----------------
def test_concat_mlp_shapes_and_backward():
    for out_dim in (1, 2):
        m = ConcatMLP((64, 32), 0.1, out_dim); xa, xb = torch.rand(8, 881), torch.rand(8, 881)
        o = m(xa, xb); assert o.shape == ((8,) if out_dim == 1 else (8, 2))
        o.sum().backward(); assert all(p.grad is not None for p in m.parameters())
    assert n_params(ConcatMLP((1024,) * 3, 0.0, 2)) == 1762 * 1024 + 1024 + 2 * (1024 * 1024 + 1024) + 1024 * 2 + 2


@pytest.mark.parametrize("inter", ["full", "sum"])
def test_siamese_symmetric_and_backward(inter):
    m = SiameseNet((32, 16), (8,), 0.3, inter).eval(); xa, xb = torch.rand(8, 881), torch.rand(8, 881)
    assert torch.allclose(m(xa, xb), m(xb, xa), atol=1e-6)            # permutation invariance
    m.train(); m(xa, xb).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_concat_is_not_symmetric():
    m = ConcatMLP((32,), 0.0, 1).eval(); xa, xb = torch.rand(8, 881), torch.rand(8, 881)
    assert not torch.allclose(m(xa, xb), m(xb, xa), atol=1e-4)


def test_checkpoint_roundtrip(tmp_path):
    m = SiameseNet((16,), (8,), 0.0, "full"); save_checkpoint(tmp_path / "c.pt", m.state_dict(), {"k": 1})
    m2 = SiameseNet((16,), (8,), 0.0, "full"); meta = load_checkpoint(tmp_path / "c.pt", m2)
    xa, xb = torch.rand(4, 881), torch.rand(4, 881)
    assert meta["k"] == 1 and torch.allclose(m.eval()(xa, xb), m2.eval()(xa, xb))


# ---------------- metrics & baselines ----------------
def test_metrics_known_values():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); s = np.array([.9, .8, .7, .2, .6, .1, .1, .1, .1, .1])
    m = metrics.compute_metrics(y, s, 0.5)
    assert m["confusion_matrix"] == [[5, 1], [1, 3]]
    assert m["precision_incompatible"] == pytest.approx(0.75) and m["recall_incompatible"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == pytest.approx((5 / 6 + 0.75) / 2)
    assert m["roc_auc"] > 0.8 and 0 < m["pr_auc"] <= 1


def test_threshold_uses_given_data_only():
    y = np.array([0, 0, 1, 1]); s = np.array([.1, .4, .6, .9]); t = metrics.best_threshold_f1(y, s)
    assert 0.4 < t <= 0.6


def test_entity_prior_unseen_gives_prior():
    m = EntityPrior(2.0).fit(np.array([1, 1, 2]), np.array([7, 8, 7]), np.array([1, 1, 0]))
    assert m.predict_score(np.array([99]), np.array([98]))[0] == pytest.approx(2 / 3, abs=1e-6)
    assert m.predict_score(np.array([1]), np.array([99]))[0] > 2 / 3


# ---------------- fingerprints (PaDEL; skipped if Java missing) ----------------
@pytest.mark.skipif(shutil.which("java") is None, reason="Java not installed")
def test_padel_pubchem_fingerprint():
    from deint import features
    smi = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)NC1=CC=C(O)C=C1"]
    fps = features.padel_fingerprints(smi)
    assert all(f is not None and f.shape == (881,) and set(np.unique(f)) <= {0, 1} for f in fps)
    assert not np.array_equal(fps[0], fps[1])
    assert features.sanity_checks({smi[0]: fps[0]})["aspirin_H_bits_ok"]
    assert np.array_equal(fps[0], features.padel_fingerprints([smi[0]])[0])      # deterministic


# ---------------- end-to-end smoke test ----------------
def test_end_to_end_smoke(tmp_path, cfg):
    from deint.experiments import Context, run_experiment
    c = copy.deepcopy(cfg)
    proc = tmp_path / "proc"; proc.mkdir()
    pairs = synth_pairs(); raw = pd.concat([pairs, pairs], ignore_index=True)[["drug_cid", "excipient_cid", "label"]]
    pairs.to_csv(proc / "pairs.csv", index=False); raw.to_csv(proc / "raw_labeled.csv", index=False)
    mols = np.unique(np.concatenate([pairs.drug_cid, pairs.excipient_cid]))
    rng = np.random.default_rng(0)
    np.savez(proc / "fingerprints.npz", fp=(rng.random((len(mols), 881)) < 0.2).astype(np.uint8), cid=mols, ok=np.ones(len(mols), bool))
    c["paths"].update(processed_dir=str(proc), results_dir=str(tmp_path / "res"))
    c["paper_protocol"].update(epochs=3, hidden=[32, 32]); c["unified_protocol"].update(epochs=3, patience=2)
    c["models"]["concat_mlp"]["hidden"] = [32, 32]
    c["models"]["siamese"].update(encoder=[32, 16], head=[8]); c["models"]["siamese_sum"].update(encoder=[32, 16], head=[8])
    c["models"]["rf"]["n_estimators"] = 10; c["models"]["concat_mlp_reg"]["hidden"] = [32, 32]
    c["seeds"] = [0, 1]
    c["experiments"]["E1_reproduction"]["seeds"] = 1; c["experiments"]["E2_leakage_audit"]["seeds"] = 1
    c["experiments"]["E3_generalization"]["seeds"] = 2; c["experiments"]["E4_ablation"]["seeds"] = 1
    c["plots"]["dpi"] = 60
    c["split"]["leakage_unit"] = "cid"            # synthetic data has no structure_groups.csv
    ctx = Context(c, torch.device("cpu"))
    for e in ["E1_reproduction", "E2_leakage_audit", "E3_generalization", "E4_ablation"]:
        run_experiment(ctx, e, progress=False)
    res = tmp_path / "res"
    must = ["config.yaml", "metrics.json", "metrics.txt", "training_log.txt", "classification_report.txt",
            "confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "training_curves.png", "predictions_test.csv"]
    d = res / "E3_generalization" / "siamese__cold_drug__seed0"
    assert all((d / f).exists() for f in must), [f for f in must if not (d / f).exists()]
    assert (d / "checkpoint_val_pr_auc.pt").exists()
    m = json.load(open(d / "metrics.json")); assert m["primary"]["test"]["n"] > 0 and m["split_report"]["test"]["frac_drug_seen_in_train"] == 0
    assert (res / "E1_reproduction" / "paper_ann__paper_rows__seed0" / "confusion_matrix_val_precision.png").exists()
    assert "full_dataset" in json.load(open(res / "E1_reproduction" / "paper_ann__paper_rows__seed0" / "metrics.json"))["primary"]
    assert (res / "E3_generalization" / "rf__cold_both__seed0" / "metrics.json").exists()
    # resume: finished runs are skipped, files untouched
    mt = (d / "metrics.json").stat().st_mtime
    run_experiment(ctx, "E3_generalization", models=["siamese"], splits=["cold_drug"], progress=False)
    assert (d / "metrics.json").stat().st_mtime == mt
    # force: old run is archived, not overwritten
    run_experiment(ctx, "E3_generalization", models=["siamese"], splits=["cold_drug"], seeds=[0], force=True, progress=False)
    assert len(list((res / "_archive").glob("siamese__cold_drug__seed0__*"))) == 1
    # aggregation
    cfg_path = tmp_path / "cfg.yaml"; yaml.safe_dump(c, open(cfg_path, "w"))
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "05_aggregate_results.py"), "--config", str(cfg_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (res / "summary" / "summary.md").exists() and (res / "summary" / "fig_E3_E4_pr_auc.png").exists()


# ---------------- structure groups / grouped splits ----------------
def test_structure_groups_merge_stereo_and_identical_fp():
    from deint.structure import build_structure_groups
    mols = pd.DataFrame({"cid": [1, 2, 3, 4], "smiles": ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CCO", "CCCO"]})
    fp = np.zeros((4, 881), dtype=np.uint8); fp[0, 1] = fp[1, 1] = 1; fp[2, 5] = 1; fp[3, 7] = 1
    g = build_structure_groups(mols, fp, np.array([1, 2, 3, 4]))
    assert g.struct_group[0] == g.struct_group[1] and len({*g.struct_group}) == 3


def test_grouped_cold_split_has_no_structure_twin_leak(cfg):
    t = synth_pairs()
    ids = sorted(set(t.drug_cid) | set(t.excipient_cid))
    groups = {c: (c if i % 5 else ids[0]) for i, c in enumerate(ids)}      # many CIDs share one structure group
    for kind in ["random_pair", "cold_drug", "cold_excipient", "cold_both"]:
        sp = splits.make_split(t, kind, 0, cfg, groups)
        assert sp["report"]["leakage_unit"] == "structure_group"
        assert sp["report"]["test"]["frac_pair_seen_in_train"] == 0
