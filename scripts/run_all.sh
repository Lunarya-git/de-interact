#!/usr/bin/env bash
# Full pipeline execution. Safe to re-run: finished stages/runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-auto}"                      # auto | cuda | cpu
SKIP_TUNING="${SKIP_TUNING:-0}"          # set SKIP_TUNING=1 to bypass Stage T
AUTO_APPROVE="${AUTO_APPROVE:-0}"        # set AUTO_APPROVE=1 to bypass interactive confirmation

run_stage() {
    local stage_name="$1"
    shift
    echo ""
    echo "=========================================================="
    echo " $stage_name"
    echo "=========================================================="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: $stage_name failed with exit code $status. Halting pipeline execution." >&2
        exit $status
    fi
}

echo "=========================================================="
echo " DE-INTERACT / DE-CrossNet Deep Learning Pipeline "
echo " Running on device: $DEVICE"
echo "=========================================================="

run_stage "Stage 1/8: Audit and clean dataset" python scripts/01_audit_and_clean.py
run_stage "Stage 2/8: Generate / verify PubChem fingerprints" python scripts/02_generate_fingerprints.py
run_stage "Stage 3/8: Build chemistry-only structure groups and splits" python scripts/03_make_splits.py
run_stage "Stage 0/8: E0 Benchmark gate (GPU throughput, VRAM, and schedule validation)" python scripts/00_benchmark.py --device "$DEVICE"

# Genuine Benchmark Gate Check
BENCH_PATH="results/E0_benchmark.json"
if [ -f "$BENCH_PATH" ]; then
    TOTAL_H=$(python -c "import json; d=json.load(open('$BENCH_PATH')); print(d.get('schedule',{}).get('total_programme_hours','?'))")
    RES_H=$(python -c "import json; d=json.load(open('$BENCH_PATH')); print(d.get('schedule',{}).get('reserve_margin_hours','?'))")
    RES_PCT=$(python -c "import json; d=json.load(open('$BENCH_PATH')); print(d.get('schedule',{}).get('reserve_percent','?'))")
    STATUS=$(python -c "import json; d=json.load(open('$BENCH_PATH')); print(d.get('schedule',{}).get('gate_status','UNKNOWN'))")

    echo ""
    echo "----------------------------------------------------------"
    echo " [GATE E0 VERDICT] Benchmark & Schedule Analysis"
    echo " Projected Runtime: $TOTAL_H hours"
    echo " Hardware Budget:   20.0 hours"
    echo " Reserve Margin:    $RES_H hours ($RES_PCT %)"
    echo " Gate Status:       $STATUS"
    echo "----------------------------------------------------------"
    echo ""

    if [ "$STATUS" != "PASS" ]; then
        echo "WARNING: Estimated execution exceeds safe budget or leaves < 15% reserve."
    fi

    if [ "$AUTO_APPROVE" != "1" ]; then
        read -r -p "Proceed with full experiment execution (Tuning and Stages 4-7)? [y/N]: " confirm
        if [[ ! "$confirm" =~ ^[yY](es)?$ ]]; then
            echo "Pipeline execution paused at Benchmark Gate E0."
            echo "Benchmark report preserved at: $BENCH_PATH"
            exit 0
        fi
        echo "User approved execution. Continuing pipeline..."
    else
        echo "AUTO_APPROVE=1 specified: proceeding automatically..."
    fi
fi

if [ "$SKIP_TUNING" != "1" ]; then
    run_stage "Stage T: Hyperparameter Optimization on Entity Pool D (Protocol T)" python scripts/06_tune.py --all-models --device "$DEVICE"
else
    echo "Stage T skipped by user flag (SKIP_TUNING=1); continuing with default configurations."
fi

run_stage "Stage 4/8: Run Experiments (E1, E2, E3, E4)" python scripts/04_run_experiments.py -e E1_reproduction E2_leakage_audit E3_generalization E4_ablation --device "$DEVICE"
run_stage "Stage 5/8: Aggregate results and generate report figures" python scripts/05_aggregate_results.py
run_stage "Stage 6/8: Sensitivity analysis E5" python scripts/07_sensitivity.py --device "$DEVICE"

if [ -f results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt ]; then
    run_stage "Stage 7/8: Attention case studies E6" python scripts/06_attention.py --checkpoint results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt --device "$DEVICE"
else
    echo "Checkpoint not found yet; skipping E6 until E3 completes."
fi

echo ""
echo "=========================================================="
echo " All pipeline stages completed successfully! "
echo " Summary report: results/summary/summary.md"
echo "=========================================================="
