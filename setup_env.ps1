<#
    VAR-Offside local setup (Windows / PowerShell).

    Creates a Python 3.10+ virtual environment, installs the package and its
    dependencies (including the 3D-replay stack), and runs the verifier.

    Usage:
        .\setup_env.ps1                 # CPU / default torch
        .\setup_env.ps1 -Cuda cu121     # install the CUDA 12.1 torch wheels
        .\setup_env.ps1 -Python "C:\Path\to\python.exe"
#>

param(
    [string]$Python = "python",
    [string]$Cuda = ""              # e.g. cu121, cu118; empty = default index
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "== VAR-Offside setup ==" -ForegroundColor Cyan

# 1. Check Python version (need 3.10+)
$verOutput = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Using Python $verOutput ($Python)"
$parts = $verOutput.Split('.')
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
    Write-Host "ERROR: Python 3.10+ is required (found $verOutput)." -ForegroundColor Red
    Write-Host "Pass a newer interpreter, e.g.  .\setup_env.ps1 -Python 'C:\Python311\python.exe'"
    exit 1
}

# 2. Create the venv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment .venv ..."
    & $Python -m venv .venv
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# 3. Upgrade pip
& $venvPy -m pip install --upgrade pip

# 4. Install PyTorch (optionally a CUDA build)
if ($Cuda -ne "") {
    Write-Host "Installing torch/torchvision for $Cuda ..." -ForegroundColor Cyan
    & $venvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
}

# 5. Install the package (editable) + dependencies
Write-Host "Installing var_offside and dependencies ..." -ForegroundColor Cyan
& $venvPy -m pip install -e .

# 6. Verify
Write-Host "Running verification ..." -ForegroundColor Cyan
& $venvPy scripts\verify_setup.py

Write-Host ""
Write-Host "Done. Activate with:  .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Launch the app with:  python -m var_offside.app" -ForegroundColor Green
