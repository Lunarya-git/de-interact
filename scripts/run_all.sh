#!/usr/bin/env bash
# Full pipeline execution. Safe to re-run: finished stages/runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-auto}"                      # auto | cuda | cpu

echo "=== Stage 1: Audit and clean ==="
python scripts/01_audit_and_clean.py

echo "=== Stage 2: PubChem fingerprints ==="
python scripts/02_generate_fingerprints.py

echo "=== Stage 3: Structure grouping and splits (chem grouping) ==="
python scripts/03_make_splits.py

echo "=== Stage 0: Benchmark and GPU check ==="
python scripts/00_benchmark.py --device "$DEVICE"

echo "=== Stage 4: Run Experiments (E1, E2, E3, E4) ==="
python scripts/04_run_experiments.py -e E1_reproduction E2_leakage_audit E3_generalization E4_ablation --device "$DEVICE"

echo "=== Stage 5: Aggregate results and generate report figures ==="
python scripts/05_aggregate_results.py

echo "=== Stage E5: Sensitivity analysis ==="
python scripts/07_sensitivity.py --device "$DEVICE"

echo "=== Stage E6: Attention case studies ==="
if [ -f results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt ]; then
    python scripts/06_attention.py --checkpoint results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt --device "$DEVICE"
fi

echo "Done! See results/summary/summary.md and figures in results/summary/."
