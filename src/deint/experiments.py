"""One experiment run = (experiment, model, split kind, seed).  Writes a self-contained run directory.

Extended in v3 to support graph-based models (DE-CrossNet, GNN Siamese) and the
independent entity-pool evaluation protocol.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from . import plots
from .baselines import EntityPrior, RFBaseline
from .metrics import best_threshold_f1, compute_metrics, metrics_text, report_text
from .models import build_model, n_params
from .splits import make_split
from .train import fit, predict_scores, save_checkpoint
from .utils import (Timer, close_logger, file_logger, now_id, prepare_run_dir, resolve, save_json,
                    set_seed)

NEURAL     = {"paper_ann", "concat_mlp", "concat_mlp_reg", "siamese", "siamese_sum"}
GNN_NEURAL = {"de_crossnet", "gnn_siamese", "de_crossnet_no_fp", "de_crossnet_self_attn", "de_crossnet_ordered_head"}
CLASSICAL  = {"rf", "entity_prior"}
ALL_MODELS = NEURAL | GNN_NEURAL | CLASSICAL


class Context:
    """Loads cached tables + fingerprints once; shared by every run.

    Also lazily loads the molecular graph cache when a graph model is requested.
    """
    def __init__(self, cfg: dict, device):
        self.cfg, self.device = cfg, device
        proc = resolve(cfg, "processed_dir"); self.proc = proc
        pairs = pd.read_csv(proc / "pairs.csv"); raw = pd.read_csv(proc / "raw_labeled.csv")
        z = np.load(proc / "fingerprints.npz", allow_pickle=True)
        cids, ok, fp = z["cid"], z["ok"], z["fp"]
        self.row = {int(c): i for i, c in enumerate(cids)}
        good = {int(c) for c in cids[ok]}
        def keep(t): return t[t.drug_cid.isin(good) & t.excipient_cid.isin(good)].reset_index(drop=True)
        self.n_dropped_pairs = len(pairs) - len(keep(pairs))
        self.pairs, self.raw = keep(pairs), keep(raw)
        # Leakage unit: cid | chem | struct | parent
        self.leakage_unit = cfg["split"].get("leakage_unit", "chem")
        self.groups = None
        if self.leakage_unit != "cid":
            sg = pd.read_csv(proc / "structure_groups.csv")
            col = {
                "chem":   "chem_group",
                "struct": "struct_group",
                "parent": "parent_group",
            }.get(self.leakage_unit, "chem_group")
            if col not in sg.columns:
                raise ValueError(f"Column '{col}' not found in structure_groups.csv. "
                                 f"Re-run script 03 to rebuild structure groups with the updated code.")
            self.groups = dict(zip(sg.cid.astype(int), sg[col].astype(int)))
        self.F_np = fp.astype(np.float32)
        self.F = torch.tensor(self.F_np, device=device)
        # Fingerprint lookup by CID (for GNN models)
        self.fp_by_cid: dict[int, int] = {int(c): i for i, c in enumerate(cids)}
        plots.set_dpi(cfg["plots"]["dpi"])

        # Lazy graph cache
        self._graph_cache = None

    @property
    def graph_cache(self):
        """Lazy-load graph cache for graph-based models."""
        if self._graph_cache is None:
            from .graphs import build_graph_cache
            mols = pd.read_csv(self.proc / "molecules.csv")
            cache, stats = build_graph_cache(mols)
            self._graph_cache = cache
            import logging
            logging.getLogger("deint.context").info(
                f"Graph cache: {stats['n_total']} molecules, "
                f"{stats['n_invalid']} invalid, "
                f"{stats['n_oversized']} oversized (>{200} atoms), "
                f"max {stats['max_atoms_seen']} atoms"
            )
        return self._graph_cache

    def table(self, split_kind: str) -> pd.DataFrame:
        return self.raw if split_kind == "paper_rows" else self.pairs

    def idx(self, t: pd.DataFrame):
        return (t.drug_cid.map(self.row).values.astype(np.int64),
                t.excipient_cid.map(self.row).values.astype(np.int64))

    def get_split(self, kind: str, seed: int) -> dict:
        d = self.proc / "splits"; d.mkdir(exist_ok=True)
        f = d / f"{kind}__{self.leakage_unit}__seed{seed}.json"
        if f.exists():
            cached = json.load(open(f))
            # Guard: reject legacy splits cached under a different leakage_unit default
            if cached.get("report", {}).get("leakage_unit", "cid") != (
                    "structure_group" if self.leakage_unit != "cid" else "cid"):
                # Stale cache: different leakage unit was used; delete and rebuild
                f.unlink()
            else:
                return cached
        sp = make_split(self.table(kind), kind, seed, self.cfg, self.groups)
        json.dump(sp, open(f, "w"))
        return sp


def _tensors(ctx, a, b, y, ix):
    dev = ctx.device
    return {"a": torch.as_tensor(a[ix], device=dev), "b": torch.as_tensor(b[ix], device=dev),
            "y": torch.as_tensor(y[ix], dtype=torch.float32, device=dev)}


def _protocol(cfg: dict, name: str):
    if name in ("paper", "paper_clean"):
        p = dict(cfg["paper_protocol"]); p["oversample"] = p["oversample_factor_clean"] if name == "paper_clean" else p["oversample_factor"]
        return p, list(p["checkpoints"]), None, "val_precision"
    p = dict(cfg["unified_protocol"]); p.setdefault("oversample", 1)
    return p, [p["checkpoint"]], p["patience"], p["checkpoint"]


def run_single(ctx: Context, exp: str, model_name: str, split_kind: str, seed: int,
               force: bool = False, progress: bool = True) -> str:
    cfg = ctx.cfg; ecfg = cfg["experiments"][exp]
    run_dir = resolve(cfg, "results_dir") / exp / f"{model_name}__{split_kind}__seed{seed}"
    if not prepare_run_dir(run_dir, force):
        return "skipped"
    timer = Timer(); run_id = now_id()
    logger = file_logger(run_dir / "training_log.txt", f"run-{run_dir}")
    try:
        proto, criteria, patience, primary = _protocol(cfg, ecfg["protocol"])
        set_seed(seed)
        table = ctx.table(split_kind); a, b = ctx.idx(table); y = table["label"].values
        sp = ctx.get_split(split_kind, seed); tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
        logger.info(f"run {run_id} | {exp} | model={model_name} split={split_kind} seed={seed} | "
                    f"train/val/test = {len(tr)}/{len(va)}/{len(te)} | positives {int(y[tr].sum())}/{int(y[va].sum())}/{int(y[te].sum())}")
        meta = {"run_id": run_id, "experiment": exp, "model": model_name, "split": split_kind, "seed": seed,
                "protocol": ecfg["protocol"], "device": str(ctx.device), "split_report": sp["report"]}
        results, hist, best_epoch, model = {}, None, {}, None
        plot_it = seed in cfg["plots"]["per_run_plot_seeds"]
        ckpt_it = seed in cfg["plots"].get("checkpoint_seeds", cfg["plots"]["per_run_plot_seeds"])

        # Check if frozen tuned hyperparameters exist from Stage T
        tuned_target = model_name
        if model_name.startswith("de_crossnet_") and not (resolve(cfg, "results_dir") / "tuned" / f"{model_name}.yaml").exists():
            tuned_target = "de_crossnet"
        tuned_candidates = [
            resolve(cfg, "results_dir") / "tuned" / f"{tuned_target}.yaml",
            Path("configs/tuned") / f"{tuned_target}.yaml",
        ]
        tuned_file = next((p for p in tuned_candidates if p.exists()), None)
        if tuned_file is not None and exp in ("E3_generalization", "E4_ablation", "E5_sensitivity"):
            try:
                with open(tuned_file) as tf:
                    tdata = yaml.safe_load(tf)
                if tdata and "best_params" in tdata:
                    bp = tdata["best_params"]
                    if "model_config" in bp and model_name in cfg.get("models", {}):
                        cfg["models"][model_name] = copy.deepcopy(cfg["models"][model_name])
                        cfg["models"][model_name].update(bp["model_config"])
                    for pk in ("lr", "weight_decay", "batch_size"):
                        if pk in bp:
                            proto[pk] = bp[pk]
                    logger.info(f"Loaded frozen tuned hyperparameters from {tuned_file}")
            except Exception as ex:
                logger.warning(f"Could not load tuned config {tuned_file}: {ex}")

        if model_name in NEURAL:
            # ── fingerprint-based neural models ──────────────────────────────────
            mcfg = {"hidden": proto["hidden"]} if model_name == "paper_ann" else cfg["models"][model_name]
            head = proto["head"]
            model = build_model(model_name, mcfg, 2 if head == "softmax2" else 1).to(ctx.device)
            meta["n_params"] = n_params(model)
            tr_fit = tr
            if proto["oversample"] > 1:
                pos = tr[y[tr] == 1]; tr_fit = np.concatenate([tr] + [pos] * (proto["oversample"] - 1))
            out = fit(model, ctx.F, _tensors(ctx, a, b, y, tr_fit), _tensors(ctx, a, b, y, va), proto, criteria,
                      ctx.device, seed, logger, desc=f"{exp[:2]} {model_name[:11]}|{split_kind[:9]}|s{seed}",
                      progress=progress, patience=patience)
            hist, best_epoch = out["history"], out["best_epoch"]
            meta.update(epochs_run=out["epochs_run"], best_epoch=best_epoch, pos_weight=out["pos_weight"],
                        train_seconds=out["train_seconds"], train_rows_after_oversampling=int(len(tr_fit)))
            pd.DataFrame(hist).to_csv(run_dir / "training_history.csv", index=False)
            for c in criteria:
                model.load_state_dict({k: v.to(ctx.device) for k, v in out["best_state"][c].items()})
                s_va = predict_scores(model, ctx.F, torch.as_tensor(a[va], device=ctx.device), torch.as_tensor(b[va], device=ctx.device), head)
                s_te = predict_scores(model, ctx.F, torch.as_tensor(a[te], device=ctx.device), torch.as_tensor(b[te], device=ctx.device), head)
                thr = 0.5 if proto["threshold"] == 0.5 else best_threshold_f1(y[va], s_va)
                r = {"val": compute_metrics(y[va], s_va, thr), "test": compute_metrics(y[te], s_te, thr),
                     "test_at_0.5": compute_metrics(y[te], s_te, 0.5)}
                if ecfg["protocol"] in ("paper", "paper_clean"):
                    s_all = predict_scores(model, ctx.F, torch.as_tensor(a, device=ctx.device), torch.as_tensor(b, device=ctx.device), head)
                    r["full_dataset"] = compute_metrics(y, s_all, thr)
                r["_scores"] = (s_te, thr); results[c] = r
                if ckpt_it:
                    save_checkpoint(run_dir / f"checkpoint_{c}.pt", out["best_state"][c],
                                    {"model": model_name, "mcfg": mcfg, "head": head, "criterion": c,
                                     "epoch": best_epoch.get(c), "seed": seed, "split": split_kind})
            if plot_it and hist:
                plots.training_curves_plot(hist, f"{model_name} | {split_kind} | seed {seed} | {exp}",
                                           run_dir / "training_curves.png", best_epoch)

        elif model_name in GNN_NEURAL:
            # ── graph-based neural models ─────────────────────────────────────────
            from .gnn import build_gnn_model, n_params as gnn_n_params
            from .gnn_train import fit_graph, predict_graph_scores

            mcfg = cfg["models"][model_name]
            head = proto["head"]
            model = build_gnn_model(model_name, mcfg).to(ctx.device)
            meta["n_params"] = gnn_n_params(model)

            # Build CID arrays for the graph trainer (not fingerprint row indices)
            cids_a = table.drug_cid.values
            cids_b = table.excipient_cid.values

            # Fingerprint matrix (None for gnn_siamese/de_crossnet_no_fp, real matrix for de_crossnet/de_crossnet_self_attn)
            use_fp = model_name in ("de_crossnet", "de_crossnet_self_attn", "de_crossnet_ordered_head")
            fp_for_gnn = ctx.F_np if use_fp else None

            tr_fit = tr
            if proto.get("oversample", 1) > 1:
                pos = tr[y[tr] == 1]; tr_fit = np.concatenate([tr] + [pos] * (proto["oversample"] - 1))

            out = fit_graph(
                model=model,
                graph_cache=ctx.graph_cache,
                train_cids_a=cids_a[tr_fit], train_cids_b=cids_b[tr_fit],
                train_y=y[tr_fit],
                val_cids_a=cids_a[va], val_cids_b=cids_b[va], val_y=y[va],
                fp_matrix=fp_for_gnn,
                proto=proto, criteria=criteria, device=ctx.device,
                seed=seed, logger=logger,
                desc=f"{exp[:2]} {model_name[:11]}|{split_kind[:9]}|s{seed}",
                progress=progress, patience=patience,
                cid_to_row=ctx.row,
            )
            hist, best_epoch = out["history"], out["best_epoch"]
            meta.update(epochs_run=out["epochs_run"], best_epoch=best_epoch,
                        pos_weight=out["pos_weight"], train_seconds=out["train_seconds"])
            pd.DataFrame(hist).to_csv(run_dir / "training_history.csv", index=False)

            gnn_bs = proto.get("val_batch_size", proto.get("batch_size", 32))
            for c in criteria:
                model.load_state_dict({k: v.to(ctx.device) for k, v in out["best_state"][c].items()})
                s_va = predict_graph_scores(model, ctx.graph_cache, cids_a[va], cids_b[va],
                                            fp_for_gnn, head, ctx.device, gnn_bs, cid_to_row=ctx.row)
                s_te = predict_graph_scores(model, ctx.graph_cache, cids_a[te], cids_b[te],
                                            fp_for_gnn, head, ctx.device, gnn_bs, cid_to_row=ctx.row)
                thr = 0.5 if proto.get("threshold") == 0.5 else best_threshold_f1(y[va], s_va)
                r = {"val": compute_metrics(y[va], s_va, thr), "test": compute_metrics(y[te], s_te, thr),
                     "test_at_0.5": compute_metrics(y[te], s_te, 0.5)}
                r["_scores"] = (s_te, thr); results[c] = r
                if ckpt_it:
                    save_checkpoint(run_dir / f"checkpoint_{c}.pt", out["best_state"][c],
                                    {"model": model_name, "mcfg": mcfg, "head": head, "criterion": c,
                                     "epoch": best_epoch.get(c), "seed": seed, "split": split_kind})
            if plot_it and hist:
                plots.training_curves_plot(hist, f"{model_name} | {split_kind} | seed {seed} | {exp}",
                                           run_dir / "training_curves.png", best_epoch)

        else:
            # ── classical baselines ───────────────────────────────────────────────
            t1 = time.time()
            d, e = table.drug_cid.values, table.excipient_cid.values
            if model_name == "rf":
                m = RFBaseline(cfg["models"]["rf"]["n_estimators"], seed).fit(ctx.F_np[a[tr]], ctx.F_np[b[tr]], y[tr])
                s_va, s_te = m.predict_score(ctx.F_np[a[va]], ctx.F_np[b[va]]), m.predict_score(ctx.F_np[a[te]], ctx.F_np[b[te]])
            else:
                m = EntityPrior(cfg["models"]["entity_prior"]["smoothing"]).fit(d[tr], e[tr], y[tr])
                s_va, s_te = m.predict_score(d[va], e[va]), m.predict_score(d[te], e[te])
            thr = best_threshold_f1(y[va], s_va)
            results[primary] = {"val": compute_metrics(y[va], s_va, thr), "test": compute_metrics(y[te], s_te, thr),
                                "test_at_0.5": compute_metrics(y[te], s_te, 0.5), "_scores": (s_te, thr)}
            meta["train_seconds"] = time.time() - t1; logger.info(f"fit {model_name} in {meta['train_seconds']:.1f}s")

        # ── write artefacts ───────────────────────────────────────────────────────
        primary_key = primary if primary in results else list(results)[0]
        blocks, txt, rep = {}, [], []
        for c, r in results.items():
            s_te, thr = r.pop("_scores")
            blocks[c] = r
            txt.append(metrics_text(f"{exp} | {model_name} | {split_kind} | seed {seed} | checkpoint={c}",
                                    {"validation": r["val"], "TEST": r["test"],
                                     **({f"full dataset (paper protocol)": r["full_dataset"]} if "full_dataset" in r else {})}))
            rep.append(f"### checkpoint={c}  (threshold {thr:.4f})\n" + report_text(y[te], s_te, thr))
            if plot_it:
                sfx = f"_{c}" if len(results) > 1 else ""
                ttl = f"{model_name} | {split_kind} | seed {seed}" + (f" | ckpt {c}" if sfx else "")
                plots.confusion_matrix_plot(r["test"]["confusion_matrix"], "Test confusion: " + ttl, run_dir / f"confusion_matrix{sfx}.png")
                plots.roc_plot(y[te], s_te, "Test ROC: " + ttl, run_dir / f"roc_curve{sfx}.png")
                plots.pr_plot(y[te], s_te, "Test PR: " + ttl, run_dir / f"precision_recall_curve{sfx}.png")
            if c == primary_key:
                pd.DataFrame({"drug_cid": table.drug_cid.values[te], "excipient_cid": table.excipient_cid.values[te],
                              "label": y[te], "score": s_te}).to_csv(run_dir / "predictions_test.csv", index=False)
        meta.update(total_seconds=timer.elapsed(), primary_checkpoint=primary_key,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_json({**meta, "results": blocks, "primary": blocks[primary_key]}, run_dir / "metrics.json")
        (run_dir / "metrics.txt").write_text("\n".join(txt))
        (run_dir / "classification_report.txt").write_text("\n".join(rep))
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump({"run": {k: meta[k] for k in ("run_id", "experiment", "model", "split", "seed", "protocol", "device")},
                            "effective_protocol": proto, "criteria": criteria,
                            "model_config": cfg["models"].get(model_name, {"hidden": proto.get("hidden")}),
                            "split_config": cfg["split"]}, f, sort_keys=False)
        return "done"
    finally:
        close_logger(logger)


def run_experiment(ctx: Context, exp: str, models=None, splits=None, seeds=None,
                   force: bool = False, progress: bool = True) -> None:
    cfg = ctx.cfg; e = cfg["experiments"][exp]
    models = models or e["models"]
    splits = splits or (e["splits"] if "splits" in e else [e["split"]])
    seeds = seeds if seeds is not None else cfg["seeds"][: e["seeds"]]
    jobs = [(m, s, k) for s in splits for m in models for k in seeds]
    log = resolve(cfg, "results_dir") / exp; log.mkdir(parents=True, exist_ok=True)
    t = Timer(); n_done = n_skip = 0
    bar = tqdm(jobs, desc=f"{exp}", unit="run", ncols=110)
    for m, s, k in bar:
        status = run_single(ctx, exp, m, s, k, force=force, progress=progress)
        n_done += status == "done"; n_skip += status == "skipped"
        bar.set_postfix(done=n_done, skipped=n_skip, elapsed=t.fmt())
    with open(log / "_experiment_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {exp} | jobs={len(jobs)} done={n_done} skipped={n_skip} | elapsed {t.fmt()}\n")
