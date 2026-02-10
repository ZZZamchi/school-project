# ROCm PyTorch install (Windows). Needs Python 3.12, AMD driver 26.1.1+
# Run: powershell -ExecutionPolicy Bypass -File setup_rocm.ps1

$ErrorActionPreference = "Stop"
$py312 = $null
try { $py312 = & py -3.12 -c "import sys; print(sys.executable)" 2>$null } catch {}
if (-not $py312) {
    Write-Host "Python 3.12 not found. ROCm needs Python 3.12."
    exit 1
}

$rocmRepo = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2"
$rocmSdk = @(
    "$rocmRepo/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl",
    "$rocmRepo/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl",
    "$rocmRepo/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl",
    "$rocmRepo/rocm-7.2.0.dev0.tar.gz"
)
$torchWheels = @(
    "$rocmRepo/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$rocmRepo/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl",
    "$rocmRepo/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl"
)

Write-Host "Using Python: $py312"
Write-Host "Installing ROCm SDK..."
& $py312 -m pip install --no-cache-dir @rocmSdk
Write-Host "Installing PyTorch (ROCm)..."
& $py312 -m pip install --no-cache-dir @torchWheels
& $py312 -m pip install transformers accelerate huggingface_hub pillow
Write-Host "Verifying GPU..."
$verifyPath = Join-Path $PSScriptRoot '_verify_gpu.py'
if (Test-Path $verifyPath) { & $py312 $verifyPath }
Write-Host "Done. Run: $py312 run_mmuavbench_official_tasks.py --max-samples 0 --models clip_vitb32 siglip_base"
exit 0
