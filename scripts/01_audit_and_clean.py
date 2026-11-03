#!/usr/bin/env python
"""Stage 1: audit the raw CSV and write the cleaned pair / molecule tables."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, save_json
from deint import data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(cfg, "processed_dir"); out.mkdir(parents=True, exist_ok=True)
    raw = data.load_raw(resolve(cfg, "raw_csv"))
    pairs, mols, raw_lab = data.clean(raw)
    a = data.audit(raw, pairs, mols)
    pairs.to_csv(out / "pairs.csv", index=False)
    mols.to_csv(out / "molecules.csv", index=False)
    raw_lab[["drug_cid", "excipient_cid", "label"]].to_csv(out / "raw_labeled.csv", index=False)
    save_json(a, out / "audit.json")
    (out / "audit.md").write_text(data.audit_markdown(a))
    print(data.audit_markdown(a))
    print(f"Wrote cleaned tables to {out}")


if __name__ == "__main__":
    main()
