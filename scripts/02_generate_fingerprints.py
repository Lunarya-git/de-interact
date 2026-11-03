#!/usr/bin/env python
"""Stage 2: PubChem fingerprints (881 bits) for every unique molecule via PaDEL. Cached & resumable."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, save_json
from deint import features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg, "processed_dir")
    mols = pd.read_csv(out / "molecules.csv")
    mat, status = features.compute_fingerprints(mols, out / "fp_cache",
                                                cfg["features"]["chunk_size"], cfg["features"]["padel_timeout"])
    n_fail = int((~status.fp_ok).sum())
    if n_fail:
        status[~status.fp_ok].to_csv(out / "fingerprint_failures.csv", index=False)
        print(f"WARNING: {n_fail} molecules failed; they (and their pairs) are dropped in stage 3.")
    status.to_csv(out / "fingerprint_status.csv", index=False)
    np.savez_compressed(out / "fingerprints.npz", fp=mat, cid=mols["cid"].values, ok=status["fp_ok"].values)
    zero = int((mat.sum(1) == 0).sum())
    rep = {"molecules": len(mols), "failed": n_fail, "all_zero_fingerprints": zero,
           "mean_bits_set": float(mat[status.fp_ok.values].sum(1).mean()),
           "distinct_fingerprints": int(len({r.tobytes() for r in mat})),
           "bits_never_set": int((mat.sum(0) == 0).sum()), "bits_always_set": int((mat.sum(0) == len(mat)).sum())}
    smi2fp = dict(zip(mols["smiles"], mat))
    rep.update(features.sanity_checks(smi2fp))
    save_json(rep, out / "fingerprint_report.json")
    print(rep)


if __name__ == "__main__":
    main()
