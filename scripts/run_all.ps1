# PowerShell Full Pipeline Runner for Windows / RTX 3050
param (
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " DE-INTERACT / DE-CrossNet Deep Learning Pipeline " -ForegroundColor Cyan
Write-Host " Running on device: $Device" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "`n[Stage 1/7] Audit and clean dataset..." -ForegroundColor Green
python scripts/01_audit_and_clean.py

Write-Host "`n[Stage 2/7] Generate / verify PubChem fingerprints..." -ForegroundColor Green
python scripts/02_generate_fingerprints.py

Write-Host "`n[Stage 3/7] Build chemistry-only structure groups and splits..." -ForegroundColor Green
python scripts/03_make_splits.py

Write-Host "`n[Stage 0/7] E0 Benchmark gate (measuring GPU throughput & VRAM)..." -ForegroundColor Green
python scripts/00_benchmark.py --device $Device

Write-Host "`n[Stage 4/7] Run Experiments E1, E2, E3, E4..." -ForegroundColor Green
python scripts/04_run_experiments.py -e E1_reproduction E2_leakage_audit E3_generalization E4_ablation --device $Device

Write-Host "`n[Stage 5/7] Aggregate experiment metrics and plots..." -ForegroundColor Green
python scripts/05_aggregate_results.py

Write-Host "`n[Stage 6/7] Run E5 Sensitivity Analyses..." -ForegroundColor Green
python scripts/07_sensitivity.py --device $Device

Write-Host "`n[Stage 7/7] Run E6 Attention Case Studies..." -ForegroundColor Green
$ckpt = "results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt"
if (Test-Path $ckpt) {
    python scripts/06_attention.py --checkpoint $ckpt --device $Device
} else {
    Write-Host "Checkpoint not found yet ($ckpt); skipping E6 until E3 is completed." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " All experiments completed successfully! " -ForegroundColor Cyan
Write-Host " Summary report: results/summary/summary.md" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

