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

    # Schedule estimate
    print("\n" + "=" * 70)
    print("ESTIMATED SCHEDULE (assuming full training at measured s/epoch):")
    schedule = []
    cfg_epochs_fp  = cfg["unified_protocol"]["epochs"]
    cfg_epochs_gnn = cfg.get("gnn_protocol", cfg["unified_protocol"])["epochs"]
    seeds_E3 = cfg["seeds"][: cfg["experiments"]["E3_generalization"]["seeds"]]
    splits_E3 = cfg["experiments"]["E3_generalization"]["splits"]
    models_E3 = cfg["experiments"]["E3_generalization"]["models"]
    for b in benchmarks:
        if "error" in b or b["s_per_epoch"] is None: continue
        m = b["model"]
        eps = cfg_epochs_gnn if m in ("de_crossnet", "gnn_siamese") else cfg_epochs_fp
        runs_E3 = sum(1 for s in splits_E3 if m in models_E3) * len(seeds_E3)
        est_h = b["s_per_epoch"] * eps * runs_E3 / 3600 * 1.1  # +10% overhead
        schedule.append({"model": m, "s_per_epoch": b["s_per_epoch"], "epochs": eps,
                         "E3_runs": runs_E3, "est_E3_hours": round(est_h, 2)})
        print(f"  {m:25s} {b['s_per_epoch']:6.1f} s/ep × {eps:3d} ep × {runs_E3:3d} runs ≈ {est_h:.1f} h")

    total_h = sum(s["est_E3_hours"] for s in schedule)
    print(f"\n  Total estimated E3 hours: {total_h:.1f} h  (20 h budget, 15–20% reserve)")
    results["schedule_estimate"] = schedule
    results["total_E3_hours_estimated"] = round(total_h, 2)

    out_path = resolve(cfg, "results_dir") / "E0_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

