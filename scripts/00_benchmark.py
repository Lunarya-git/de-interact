#!/usr/bin/env python
"""Stage 0: GPU / CPU benchmark gate (E0).

Measures actual runtime and GPU memory for one epoch of every model class
on the real data (one seed, random_pair split, no full training).
Outputs a JSON report that the experiment schedule uses to calibrate timings.

Run BEFORE launching the full experiment suite:
  python scripts/00_benchmark.py [--device cuda] [--config configs/default.yaml]
"""
import argparse, json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, pick_device, save_json


def gpu_mem_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def benchmark_fp_model(name, cfg, device, ctx, split, seed, n_epochs=3, progress=True):
    """Benchmark a fingerprint-based model: train n_epochs, measure time and peak mem."""
    from deint.experiments import _protocol, _tensors
    from deint.models import build_model, n_params
    from deint.train import fit
    import pandas as pd

    table = ctx.table(split)
    a, b = ctx.idx(table); y = table["label"].values
    sp = ctx.get_split(split, seed); tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    ecfg = cfg["experiments"]["E0_benchmark"]
    proto, criteria, patience, primary = _protocol(cfg, ecfg["protocol"])
    proto = dict(proto); proto["epochs"] = n_epochs; proto.pop("patience", None)

    mcfg = {"hidden": [64, 64, 64]} if name == "paper_ann" else cfg["models"].get(name, {})
    head = proto["head"]
    model = build_model(name, mcfg, 2 if head == "softmax2" else 1).to(device)

    reset_mem()
    t0 = time.time()
    out = fit(model, ctx.F, _tensors(ctx, a, b, y, tr), _tensors(ctx, a, b, y, va),
              proto, criteria, device, seed, logger=None, desc=f"bench {name}",
              progress=progress, patience=None)
    elapsed = time.time() - t0
    peak_mb = gpu_mem_mb()
    return {
        "model": name, "n_params": n_params(model), "n_train": len(tr),
        "n_epochs_run": n_epochs, "elapsed_s": round(elapsed, 2),
        "s_per_epoch": round(elapsed / n_epochs, 2), "peak_gpu_mb": round(peak_mb, 1),
    }


def benchmark_gnn_model(name, cfg, device, ctx, split, seed, n_epochs=2, progress=True):
    """Benchmark a GNN model: 2 epochs, measure time and peak VRAM."""
    from deint.gnn import build_gnn_model, n_params
    from deint.gnn_train import fit_graph
    import pandas as pd

    table = ctx.table(split)
    a, b = ctx.idx(table); y = table["label"].values
    sp = ctx.get_split(split, seed); tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))

    gnn_proto = dict(cfg.get("gnn_protocol", cfg["unified_protocol"]))
    gnn_proto["epochs"] = n_epochs

    mcfg = cfg["models"][name]
    model = build_gnn_model(name, mcfg).to(device)

    cids_a = table.drug_cid.values; cids_b = table.excipient_cid.values
    use_fp = (name == "de_crossnet")
    fp_for_gnn = ctx.F_np if use_fp else None

    reset_mem()
    t0 = time.time()
    out = fit_graph(
        model=model,
        graph_cache=ctx.graph_cache,
        train_cids_a=cids_a[tr], train_cids_b=cids_b[tr], train_y=y[tr],
        val_cids_a=cids_a[va], val_cids_b=cids_b[va], val_y=y[va],
        fp_matrix=fp_for_gnn,
        proto=gnn_proto, criteria=[gnn_proto["checkpoint"]],
        device=device, seed=seed, logger=None,
        desc=f"bench {name}", progress=progress, patience=None,
        cid_to_row=ctx.row,
    )
    elapsed = time.time() - t0
    peak_mb = gpu_mem_mb()
    return {
        "model": name, "n_params": n_params(model), "n_train": len(tr),
        "n_epochs_run": n_epochs, "elapsed_s": round(elapsed, 2),
        "s_per_epoch": round(elapsed / n_epochs, 2), "peak_gpu_mb": round(peak_mb, 1),
    }


def benchmark_classical(name, cfg, device, ctx, split, seed):
    from deint.experiments import _protocol
    from deint.baselines import EntityPrior, RFBaseline

    table = ctx.table(split)
    a, b = ctx.idx(table); y = table["label"].values
    sp = ctx.get_split(split, seed); tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
    d, e = table.drug_cid.values, table.excipient_cid.values

    t0 = time.time()
    if name == "rf":
        m = RFBaseline(50, seed).fit(ctx.F_np[a[tr]], ctx.F_np[b[tr]], y[tr])
    else:
        m = EntityPrior(2.0).fit(d[tr], e[tr], y[tr])
    elapsed = time.time() - t0
    return {"model": name, "n_params": 0, "n_train": len(tr),
            "elapsed_s": round(elapsed, 2), "s_per_epoch": None, "peak_gpu_mb": 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--config", default=None)
    ap.add_argument("--fp-epochs", type=int, default=3)
    ap.add_argument("--gnn-epochs", type=int, default=2)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from deint.experiments import Context
    ctx = Context(cfg, device)
    print(f"Pairs: {len(ctx.pairs)}  |  FP matrix: {ctx.F.shape}  |  dropped: {ctx.n_dropped_pairs}")

    results = {"device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"}
    benchmarks = []
    split, seed = "random_pair", 0
    progress = not args.no_progress

    # FP models (quick benchmark)
    for name in ["concat_mlp", "siamese"]:
        print(f"\nBenchmarking {name} ({args.fp_epochs} epochs)...", flush=True)
        try:
            r = benchmark_fp_model(name, cfg, device, ctx, split, seed, args.fp_epochs, progress)
            benchmarks.append(r)
            print(f"  {r['s_per_epoch']:.1f} s/epoch  |  {r['peak_gpu_mb']:.0f} MB  |  {r['n_params']:,} params")
        except Exception as ex:
            print(f"  FAILED: {ex}")
            benchmarks.append({"model": name, "error": str(ex)})

    # GNN models
    from deint.gnn import build_gnn_model
    for name in ["gnn_siamese", "de_crossnet"]:
        print(f"\nBenchmarking {name} ({args.gnn_epochs} epochs)...", flush=True)
        try:
            r = benchmark_gnn_model(name, cfg, device, ctx, split, seed, args.gnn_epochs, progress)
            benchmarks.append(r)
            print(f"  {r['s_per_epoch']:.1f} s/epoch  |  {r['peak_gpu_mb']:.0f} MB  |  {r['n_params']:,} params")
        except Exception as ex:
            print(f"  FAILED: {ex}")
            benchmarks.append({"model": name, "error": str(ex)})

    # Classical (no epochs)
    for name in ["rf", "entity_prior"]:
        print(f"\nBenchmarking {name}...", flush=True)
        try:
            r = benchmark_classical(name, cfg, device, ctx, split, seed)
            benchmarks.append(r)
            print(f"  fit time: {r['elapsed_s']:.1f} s")
        except Exception as ex:
            print(f"  FAILED: {ex}")
            benchmarks.append({"model": name, "error": str(ex)})

    results["benchmarks"] = benchmarks

    # Schedule estimate across complete experiment programme
    print("\n" + "=" * 70)
    print("CONSERVATIVE SCHEDULE PROJECTION ACROSS FULL EXPERIMENT PROGRAMME:")
    print("=" * 70)
    
    cfg_epochs_fp  = cfg["unified_protocol"]["epochs"]
    cfg_epochs_gnn = cfg.get("gnn_protocol", cfg["unified_protocol"])["epochs"]
    
    # Map benchmarked speeds (s/run)
    bench_map = {b["model"]: b for b in benchmarks if "error" not in b}
    def est_run_seconds(m):
        if m in bench_map:
            b = bench_map[m]
            if b["s_per_epoch"] is not None:
                eps = cfg_epochs_gnn if m in ("de_crossnet", "gnn_siamese", "de_crossnet_no_fp", "de_crossnet_self_attn", "de_crossnet_ordered_head") else cfg_epochs_fp
                return b["s_per_epoch"] * eps
            return b["elapsed_s"]
        # Fallback to closest benchmarked model
        if "crossnet" in m or "gnn" in m:
            ref = bench_map.get("de_crossnet", bench_map.get("gnn_siamese", {}))
            return (ref.get("s_per_epoch", 2.0)) * cfg_epochs_gnn
        if "siamese" in m:
            ref = bench_map.get("siamese", {})
            return (ref.get("s_per_epoch", 0.5)) * cfg_epochs_fp
        ref = bench_map.get("concat_mlp", {})
        return (ref.get("s_per_epoch", 0.5)) * cfg_epochs_fp

    exp_schedule = {}
    total_sec = 0.0

    # ── Stage T: Hyperparameter Tuning on Entity Pool D (Protocol T) ──────
    # 125 Optuna trials across 4 models: de_crossnet (40), gnn_siamese (30), concat_mlp_reg (30), siamese (25)
    # Multi-split tuning over 3 splits x 2 seeds = 6 conditions inside Pool D
    # Pool D contains ~2,031 pairs (~56% of total pairs).
    # With early-stopping and Optuna median pruning, trials prune early (~3.5 trainings per trial -> ~437 expected trainings)
    # Wall-clock caps from §15: de_crossnet: 2.2h, gnn_siamese: 0.9h, concat_mlp_reg: 0.45h, siamese: 0.40h (sum: 3.95h)
    tuning_budgets = {
        "de_crossnet":    {"trials": 40, "cap_h": 2.2,  "epochs": 22},
        "gnn_siamese":    {"trials": 30, "cap_h": 0.9,  "epochs": 22},
        "concat_mlp_reg": {"trials": 30, "cap_h": 0.45, "epochs": 30},
        "siamese":        {"trials": 25, "cap_h": 0.40, "epochs": 30},
    }
    stage_t_sec = 0.0
    stage_t_breakdown = {}
    for tm, tinfo in tuning_budgets.items():
        base_s_ep = bench_map.get(tm, {}).get("s_per_epoch")
        if base_s_ep is None:
            base_s_ep = 4.0 if "crossnet" in tm else (2.0 if "gnn" in tm else 0.5)
        # Expected trainings: trials * 3.5 with median pruner
        exp_trainings = tinfo["trials"] * 3.5
        # Training in Pool D takes ~0.56x full epoch time due to smaller pool size
        est_sec = exp_trainings * (base_s_ep * 0.56 * tinfo["epochs"]) * 1.10
        model_h = min(est_sec / 3600.0, tinfo["cap_h"])
        stage_t_breakdown[tm] = {
            "trials": tinfo["trials"],
            "expected_trainings": int(exp_trainings),
            "estimated_hours": round(model_h, 2),
            "cap_hours": tinfo["cap_h"]
        }
        stage_t_sec += model_h * 3600.0

    stage_t_h = stage_t_sec / 3600.0
    exp_schedule["Stage_T_tuning"] = {
        "total_trials": 125,
        "expected_trainings": 437,
        "estimated_hours": round(stage_t_h, 2),
        "wall_clock_cap_hours": 3.95,
        "models": stage_t_breakdown
    }
    total_sec += stage_t_sec
    print(f"  {'Stage_T_tuning':20s}: 125 trials (~437 trainings) -> ~{stage_t_h:5.2f} hours (cap: 3.95 h)")

    for exp_id in ["E1_reproduction", "E2_leakage_audit", "E3_generalization", "E4_ablation"]:
        if exp_id not in cfg["experiments"]:
            continue
        ecfg = cfg["experiments"][exp_id]
        splits = ecfg.get("splits", [ecfg.get("split", "random_pair")])
        models = ecfg.get("models", [])
        seeds = cfg["seeds"][: ecfg.get("seeds", 1)]
        exp_runs = len(splits) * len(models) * len(seeds)
        exp_sec = sum(est_run_seconds(m) for m in models) * len(splits) * len(seeds)
        exp_h = (exp_sec / 3600.0) * 1.15  # +15% validation/checkpoint overhead
        exp_schedule[exp_id] = {
            "splits": len(splits), "models": len(models), "seeds": len(seeds),
            "total_runs": exp_runs, "estimated_hours": round(exp_h, 2)
        }
        total_sec += exp_sec * 1.15
        print(f"  {exp_id:20s}: {exp_runs:3d} runs  ->  ~{exp_h:5.2f} hours")

    # E5 sensitivity estimate
    if "E5_sensitivity" in cfg["experiments"]:
        e5_cfg = cfg["experiments"]["E5_sensitivity"]
        e5_units = len(e5_cfg.get("leakage_units", ["chem", "struct", "parent"]))
        e5_splits = len(e5_cfg.get("splits", ["random_pair", "cold_drug"]))
        e5_models = e5_cfg.get("models", ["de_crossnet", "siamese"])
        e5_seeds = len(cfg["seeds"][: e5_cfg.get("seeds", 1)])
        e5_runs = e5_units * e5_splits * len(e5_models) * e5_seeds
        e5_sec = sum(est_run_seconds(m) for m in e5_models) * e5_units * e5_splits * e5_seeds * 1.15
        e5_h = e5_sec / 3600.0
        exp_schedule["E5_sensitivity"] = {"total_runs": e5_runs, "estimated_hours": round(e5_h, 2)}
        total_sec += e5_sec
        print(f"  {'E5_sensitivity':20s}: {e5_runs:3d} runs  ->  ~{e5_h:5.2f} hours")

    # E6 attention estimate (inference on ~10 pairs)
    e6_h = 0.05
    exp_schedule["E6_attention"] = {"estimated_hours": e6_h}
    total_sec += e6_h * 3600.0
    print(f"  {'E6_attention':20s}: inference only ->  ~{e6_h:5.2f} hours")

    total_programme_hours = total_sec / 3600.0
    budget_hours = 20.0
    reserve_hours = budget_hours - total_programme_hours
    reserve_pct = (reserve_hours / budget_hours) * 100.0
    gate_pass = (reserve_pct >= 15.0 and total_programme_hours <= budget_hours)

    print("-" * 70)
    print(f"  Total Estimated Programme Runtime: {total_programme_hours:5.2f} hours")
    print(f"  Available Hardware Budget:          {budget_hours:5.2f} hours")
    print(f"  Reserve Safety Margin:              {reserve_hours:5.2f} hours ({reserve_pct:.1f}%)")
    print(f"  Benchmark Gate Verdict:             {'PASS (Ready to proceed)' if gate_pass else 'WARNING (Budget tight)'}")
    print("=" * 70)

    results["schedule"] = {
        "experiments": exp_schedule,
        "total_programme_hours": round(total_programme_hours, 2),
        "budget_hours": budget_hours,
        "reserve_margin_hours": round(reserve_hours, 2),
        "reserve_percent": round(reserve_pct, 1),
        "gate_status": "PASS" if gate_pass else "WARNING"
    }
    results["total_E3_hours_estimated"] = round(exp_schedule.get("E3_generalization", {}).get("estimated_hours", 0.0), 2)

    out_path = resolve(cfg, "results_dir") / "E0_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f"\nBenchmark gate report written to {out_path}\n")


if __name__ == "__main__":
    main()

