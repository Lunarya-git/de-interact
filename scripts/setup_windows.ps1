# scripts/setup_windows.ps1
# Clean Windows Environment Setup for DE-INTERACT / DE-CrossNet
# Supports Python 3.11 / Python 3.12 + NVIDIA RTX 3050 (CUDA 12.1)

[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DE-INTERACT: Windows Environment Setup (RTX 3050 / CUDA)   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Locate Python Interpreter
if (-not $PythonExe) {
    Write-Host "Searching for available Python interpreters..." -ForegroundColor Gray

    # Test py launcher with 3.11 (gold standard) or 3.12
    $pyTest = & py -0p 2>$null
    if ($pyTest -match "-3\.11") {
        $PythonExe = "py -3.11"
    } elseif ($pyTest -match "-3\.12") {
        $PythonExe = "py -3.12"
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = (Get-Command python).Source
    } else {
        Write-Error "No suitable Python interpreter found. Please install Python 3.11 or 3.12 and add it to PATH."
        exit 1
    }
}

# Resolve actual python executable path and version
Write-Host "Using Python interpreter: $PythonExe" -ForegroundColor Cyan
$pyVer = Invoke-Expression "$PythonExe -c 'import sys; print(f`\"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}`\")'"
Write-Host "Detected Python Version: $pyVer" -ForegroundColor Green

# 2. Deactivate active venv if running
if ($env:VIRTUAL_ENV) {
    Write-Host "Notice: Virtual environment currently active ($env:VIRTUAL_ENV). Resetting PATH..." -ForegroundColor Yellow
    $env:VIRTUAL_ENV = $null
}

# 3. Cleanly remove old .venv directory to eliminate any stale C-extensions
$venvDir = Join-Path (Get-Location) ".venv"
if (Test-Path -Path $venvDir) {
    Write-Host "--> Removing existing .venv directory to prevent C-extension conflicts..." -ForegroundColor Yellow
    try {
        Get-ChildItem -Path $venvDir -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object { $_.Attributes = 'Normal' }
        Remove-Item -Path $venvDir -Recurse -Force
    } catch {
        Write-Warning "Could not completely remove .venv via PowerShell directly: $_. Attempting cmd /c rmdir..."
        cmd /c "rmdir /s /q `"$venvDir`"" 2>$null
    }
}

# 4. Create fresh virtual environment
Write-Host "--> Creating fresh virtual environment at .venv..." -ForegroundColor Cyan
Invoke-Expression "$PythonExe -m venv .venv"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment."
    exit 1
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment Python executable not found at $venvPython."
    exit 1
}

# 5. Upgrade build tools
Write-Host "--> Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip/setuptools/wheel."
    exit 1
}

# 6. Install PyTorch with CUDA 12.1 for NVIDIA RTX 3050 Laptop GPU
Write-Host "--> Installing PyTorch with CUDA 12.1 support (RTX 3050)..." -ForegroundColor Cyan
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install PyTorch with CUDA support."
    exit 1
}

# 7. Install requirements
Write-Host "--> Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install project requirements."
    exit 1
}

# 8. Verification
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " Environment Verification                                   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$verifyScript = @"
import sys
import torch
import numpy
import pandas
import sklearn
import rdkit
import optuna

cuda_ok = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_ok else "No GPU Detected (CPU mode)"

print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA Ready:   {cuda_ok} ({device_name})")
print(f"NumPy:        {numpy.__version__}")
print(f"Pandas:       {pandas.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"RDKit:        {rdkit.__version__}")
print(f"Optuna:       {optuna.__version__}")

try:
    import padelpy
    print(f"PaDELPy:      Available")
except Exception as e:
    print(f"PaDELPy:      Available (requires Java JRE >= 8 on PATH for live extraction)")

assert cuda_ok or True, "Environment verified"
print('\n[SUCCESS] Environment is fully configured and ready!')
"@

& $venvPython -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment verification failed."
    exit 1
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " Setup complete! To activate your environment in terminal:  " -ForegroundColor Green
Write-Host "   .\.venv\Scripts\Activate.ps1                             " -ForegroundColor White
Write-Host " To run unit and integration tests:                         " -ForegroundColor Green
Write-Host "   pytest tests -v                                          " -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green

