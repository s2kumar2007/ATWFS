param(
    [switch]$Synthetic = $false,
    [string]$Config = "config.yaml"
)

$ErrorActionPreference = "Stop"
Write-Host "=== ATWFS Reproducibility Pipeline ===" -ForegroundColor Cyan

$SynFlag = ""
if ($Synthetic) {
    $SynFlag = "--synthetic"
    Write-Host "Running in SYNTHETIC mode" -ForegroundColor Yellow
}

Write-Host "[1/7] Checking / Downloading Weights..."
python scripts/download_weights.py --config $Config

Write-Host "[2/7] Stage 1-4: Training Teacher..."
if ($Synthetic) {
    python -m training.train --config $Config --synthetic --cpu-fallback
} else {
    python -m training.train --config $Config
}

Write-Host "[3/7] Stage 5: Distilling Student..."
if ($Synthetic) {
    python -m training.distill --config $Config --synthetic
} else {
    python -m training.distill --config $Config
}

Write-Host "[4/7] Stage 6: Compression & Benchmark (INT8 PTQ + Structured Pruning)..."
if ($Synthetic) {
    python -m evaluation.benchmark --config $Config --synthetic
} else {
    python -m evaluation.benchmark --config $Config
}

Write-Host "[5/7] Stage 9: Training FusionNet & Ablation Report..."
python -m training.fusion_train --config $Config

Write-Host "[6/7] Stage 11: IDD Protocol Evaluation..."
$Ckpt = "outputs/checkpoints/student_best.pt"
if (-not (Test-Path $Ckpt)) {
    $Ckpt = "outputs/checkpoints/teacher_best.pt"
}
if ($Synthetic) {
    python -m evaluation.evaluate --config $Config --checkpoint $Ckpt --synthetic
} else {
    python -m evaluation.evaluate --config $Config --checkpoint $Ckpt
}

Write-Host "[7/7] Running Smoke Tests..."
python run_all_smoke_tests.py

Write-Host "=== Reproducibility Pipeline Complete ===" -ForegroundColor Green
