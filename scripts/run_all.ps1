# PowerShell Full Pipeline Runner for Windows / RTX 3050
param (
    [string]$Device = "auto",
    [switch]$AutoApprove = $false,
    [switch]$SkipTuning = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $ScriptDir "..")

function Invoke-Stage {
    param (
        [string]$StageName,
        [scriptblock]$Command
    )
    Write-Host "`n==========================================================" -ForegroundColor Cyan
    Write-Host " $StageName" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[FATAL ERROR] $StageName failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "Pipeline execution stopped immediately to prevent cascading errors." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " DE-INTERACT / DE-CrossNet Deep Learning Pipeline " -ForegroundColor Cyan
Write-Host " Running on device: $Device" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Invoke-Stage "Stage 1/8: Audit and clean dataset" {
    python scripts/01_audit_and_clean.py
}

Invoke-Stage "Stage 2/8: Generate / verify PubChem fingerprints" {
    python scripts/02_generate_fingerprints.py
}

Invoke-Stage "Stage 3/8: Build chemistry-only structure groups and splits" {
    python scripts/03_make_splits.py
}

Invoke-Stage "Stage 0/8: E0 Benchmark gate (GPU throughput, VRAM, and schedule validation)" {
    python scripts/00_benchmark.py --device $Device
}

# Genuine Benchmark Gate Check
$benchPath = "results/E0_benchmark.json"
if (Test-Path $benchPath) {
    $benchJson = Get-Content $benchPath -Raw | ConvertFrom-Json
    $sched = $benchJson.schedule
    $totalHours = $sched.total_programme_hours
    $reserveHours = $sched.reserve_margin_hours
    $reservePct = $sched.reserve_percent
    $verdict = $sched.gate_status

    Write-Host "`n----------------------------------------------------------" -ForegroundColor Yellow
    Write-Host " [GATE E0 VERDICT] Benchmark & Schedule Analysis" -ForegroundColor Yellow
    Write-Host " Projected Runtime: $totalHours hours" -ForegroundColor Yellow
    Write-Host " Hardware Budget:   20.0 hours" -ForegroundColor Yellow
    Write-Host " Reserve Margin:    $reserveHours hours ($reservePct %)" -ForegroundColor Yellow
    Write-Host " Gate Status:       $verdict" -ForegroundColor $(if ($verdict -eq "PASS") { "Green" } else { "Red" })
    Write-Host "----------------------------------------------------------`n" -ForegroundColor Yellow

    if ($verdict -ne "PASS") {
        Write-Host "WARNING: Estimated execution exceeds safe budget or leaves < 15% reserve." -ForegroundColor Red
        Write-Host "Consider reducing seeds or ablation models in configs/default.yaml before continuing." -ForegroundColor Red
    }

    if (-not $AutoApprove) {
        $confirm = Read-Host "Proceed with full experiment execution (Tuning and Stages 4-7)? [y/N]"
        if ($confirm -notmatch "^[yY](es)?$") {
            Write-Host "`nPipeline execution paused at Benchmark Gate E0." -ForegroundColor Cyan
            Write-Host "Benchmark metrics preserved at: $benchPath" -ForegroundColor Cyan
            Write-Host "To adjust configuration, edit configs/default.yaml. When ready, rerun or run subsequent stages directly." -ForegroundColor Cyan
            exit 0
        }
        Write-Host "User approved execution. Continuing pipeline..." -ForegroundColor Green
    } else {
        Write-Host "AutoApprove flag specified: proceeding automatically..." -ForegroundColor Green
    }
}

if (-not $SkipTuning) {
    Invoke-Stage "Stage T: Hyperparameter Optimization on Entity Pool D (Protocol T)" {
        python scripts/06_tune.py --all-models --device $Device
    }
} else {
    Write-Host "`n[Stage T: Skipped by user flag - using default configurations]" -ForegroundColor Yellow
}

Invoke-Stage "Stage 4/8: Run Experiments E1, E2, E3, E4" {
    python scripts/04_run_experiments.py -e E1_reproduction E2_leakage_audit E3_generalization E4_ablation --device $Device
}

Invoke-Stage "Stage 5/8: Aggregate experiment metrics and plots" {
    python scripts/05_aggregate_results.py
}

Invoke-Stage "Stage 6/8: Run E5 Sensitivity Analyses" {
    python scripts/07_sensitivity.py --device $Device
}

$ckpt = "results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt"
if (Test-Path $ckpt) {
    Invoke-Stage "Stage 7/8: Run E6 Attention Case Studies" {
        python scripts/06_attention.py --checkpoint $ckpt --device $Device
    }
} else {
    Write-Host "`n[Stage 7/8: Checkpoint not found yet ($ckpt); skipping E6 until E3 completes]" -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " All pipeline stages completed successfully! " -ForegroundColor Cyan
Write-Host " Summary report: results/summary/summary.md" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

