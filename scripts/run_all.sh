#!/usr/bin/env bash
# Full pipeline.  Safe to re-run: finished stages/runs are skipped (cached fingerprints, finished run dirs).
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-auto}"                      # auto | cuda | cpu
python scripts/01_audit_and_clean.py
python scripts/02_generate_fingerprints.py
python scripts/03_make_splits.py
python scripts/04_run_experiments.py -e all --device "$DEVICE"
python scripts/05_aggregate_results.py
echo "Done. See results/summary/summary.md"
