#!/usr/bin/env python
"""Stage 5: collect every metrics.json into summary tables and comparison figures."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve
from deint import plots

M = ["pr_auc", "roc_auc", "macro_f1", "f1_incompatible", "recall_incompatible", "precision_incompatible",
     "balanced_accuracy", "mcc", "accuracy"]
PAPER_REF = {"ref": "Paper (max-precision ckpt, FULL dataset)", "accuracy": (3493 + 687) / 4248,
             "precision_incompatible": 0.92, "recall_incompatible": 0.98, "f1_incompatible": 0.95}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def pm(x): return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}" if len(x) > 1 else f"{x.iloc[0]:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config); plots.set_dpi(cfg["plots"]["dpi"])
    res = resolve(cfg, "results_dir"); out = res / "summary"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(res.glob("E*/*/metrics.json")):
        m = json.load(open(f))
        for ck, r in m["results"].items():
            row = {"exp": m["experiment"], "model": m["model"], "split": m["split"], "seed": m["seed"], "ckpt": ck,
                   "is_primary": ck == m["primary_checkpoint"], "train_s": m.get("train_seconds"), "epochs": m.get("epochs_run"),
                   "n_test": r["test"]["n"], "n_test_pos": r["test"]["n_pos"]}
            row.update({k: r["test"][k] for k in M})
            if "full_dataset" in r: row.update({f"full_{k}": r["full_dataset"][k] for k in M})
            rows.append(row)
    if not rows:
        sys.exit("No results found - run stage 4 first.")
    df = pd.DataFrame(rows); df.to_csv(out / "all_runs.csv", index=False)
    L = ["# Results summary (auto-generated; mean ± std over seeds; TEST partitions)", ""]

    pap = df[df.exp.isin(["E1_reproduction", "E2_leakage_audit"])]
    if len(pap):
        t = pap.groupby(["exp", "ckpt"]).agg(**{k: (k, pm) for k in ["accuracy", "precision_incompatible", "recall_incompatible", "f1_incompatible", "macro_f1", "pr_auc"]},
                                            full_acc=("full_accuracy", pm), full_prec=("full_precision_incompatible", pm),
                                            full_rec=("full_recall_incompatible", pm), n=("seed", "size")).reset_index()
        L += ["## E1 / E2 - paper protocol: reproduction vs. duplicate-free evaluation", "",
              "`full_*` = metrics on the whole dataset (the paper's reporting protocol; contains training rows).", "",
              md(t), "",
              f"Paper reference (max-precision checkpoint, full dataset): accuracy {PAPER_REF['accuracy']:.3f}, "
              f"incompatible precision {PAPER_REF['precision_incompatible']}, recall {PAPER_REF['recall_incompatible']}, "
              f"F1 {PAPER_REF['f1_incompatible']}.", ""]
        p = pap[pap.ckpt == "val_precision"].melt(id_vars=["exp"], value_vars=["precision_incompatible", "recall_incompatible", "f1_incompatible", "pr_auc"], var_name="metric")
        g = p.groupby(["metric", "exp"]).value.agg(["mean", "std"]).reset_index().fillna(0)
        plots.grouped_bar(g, "metric", "exp", "mean", "std", "Paper-protocol ANN: leaky row split (E1) vs duplicate-free pair split (E2), test set",
                          "Score", out / "fig_E1_vs_E2_leakage.png", ylim=(0, 1.05))

    gen = df[df.exp.isin(["E3_generalization", "E4_ablation"]) & df.is_primary]
    if len(gen):
        order = ["entity_prior", "rf", "concat_mlp", "concat_mlp_reg", "siamese_sum", "siamese",
                 "gnn_siamese", "de_crossnet_no_fp", "de_crossnet_self_attn", "de_crossnet"]
        sp_order = [s for s in ["random_pair", "cold_drug", "cold_excipient", "cold_both"] if s in set(gen.split)]
        t = gen.groupby(["split", "model"]).agg(**{k: (k, pm) for k in M}, n_seeds=("seed", "size"),
                                                avg_test_pos=("n_test_pos", "mean")).reset_index()
        t["_s"] = t.split.map(sp_order.index); t["_m"] = t.model.map(lambda x: order.index(x) if x in order else 99)
        t = t.sort_values(["_s", "_m"]).drop(columns=["_s", "_m"])
        L += ["## E3 / E4 - generalisation across split types (unified protocol)", "", md(t.round(1)), ""]
        # paired differences (same split + seed => same data partition)
        piv = gen.pivot_table(index=["split", "seed"], columns="model", values=["pr_auc", "macro_f1"])
        pr = []
        for ref in ["concat_mlp", "concat_mlp_reg", "siamese_sum", "rf", "entity_prior"]:
            for s in sp_order:
                for met in ["pr_auc", "macro_f1"]:
                    if ("siamese" in piv[met].columns) and (ref in piv[met].columns):
                        d = (piv[met]["siamese"] - piv[met][ref]).xs(s, level="split").dropna()
                        if len(d): pr.append({"split": s, "siamese minus": ref, "metric": met, "mean_diff": f"{d.mean():+.3f}",
                                              "std": f"{d.std(ddof=0):.3f}", "siamese_wins": f"{int((d > 0).sum())}/{len(d)}"})
        if pr:
            L += ["### Paired differences (proposed siamese − reference), per seed", "", md(pd.DataFrame(pr)), ""]
            pd.DataFrame(pr).to_csv(out / "paired_differences.csv", index=False)
        t.to_csv(out / "E3_E4_summary.csv", index=False)
        for met, nm in [("pr_auc", "PR-AUC (incompatible = positive)"), ("macro_f1", "Macro F1")]:
            g = gen.groupby(["split", "model"])[met].agg(["mean", "std"]).reset_index().fillna(0)
            g["model"] = pd.Categorical(g.model, [m for m in order if m in set(g.model)], ordered=True)
            g = g.sort_values(["model"])
            g["split"] = pd.Categorical(g.split, sp_order, ordered=True); g = g.sort_values(["split", "model"])
            plots.grouped_bar(g.astype({"model": str, "split": str}), "split", "model", "mean", "std",
                              f"Test {nm} by split type (mean ± std over seeds)", nm, out / f"fig_E3_E4_{met}.png",
                              ylim=(0, 1.05))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L)); print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
