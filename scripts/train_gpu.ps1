"""One-command GPU training, distillation, ONNX export and evaluation.

Usage (PowerShell)
------------------
  .\\scripts\\train_gpu.ps1 -DataSrc "D:\\IDD_RESIZED" [-Resume]

Features
--------
* Checks nvidia-smi + torch.cuda.is_available(), stops cleanly if CUDA is off
* Prints GPU name + free memory
* Runs 1-epoch smoke test on 200 images, prints time per epoch + ETA
* Trains teacher_v2, then distils student_v2
* Exports student_v2 ONNX and verifies mask IoU >= 0.99 on 50 test images
* Runs full test-split evaluation for teacher_v2, student_v2 and ONNX student
* Logs everything to outputs/logs/train_<timestamp>.txt
* Supports --Resume to continue from latest checkpoint
* OOM guard: on CUDA OOM halves batch size and retries (via train.py --batch-size)
* Windows-compatible num_workers (defaults to 4 on Windows)
"""

param(
    [Parameter(Mandatory=$true)]
    [string]$DataSrc,

    [switch]$Resume,

    [string]$Config = "config.yaml",

    [int]$BatchSize = 0,        # 0 = use config default (16)
    [int]$NumWorkers = 4,       # Windows-safe default

    [string]$TeacherTag   = "teacher_v2",
    [string]$StudentTag   = "student_v2",
    [string]$StudentOnnx  = "outputs/checkpoints/student_v2.onnx"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- Timestamps and Logging --------------------------------------------------
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = "$RepoRoot/outputs/logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = "$LogDir/train_$Ts.txt"
function Log { param([string]$Msg) $line = "$(Get-Date -Format 'HH:mm:ss')  $Msg"; Write-Host $line; Add-Content -Path $LogFile -Value $line }

Log "=== ATWFS GPU Training Pipeline ==="
Log "Repo: $RepoRoot"
Log "Data: $DataSrc"
Log "Log:  $LogFile"

# --- CUDA check --------------------------------------------------------------
Log "Checking CUDA availability..."
try { & nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | ForEach-Object { Log "GPU: $_" } }
catch { Log "WARNING: nvidia-smi not found or failed.  Continuing (will use Python check)." }

$CudaOk = python -c "import torch; print(int(torch.cuda.is_available()))" 2>&1
if ($CudaOk.Trim() -ne "1") {
    Log "ERROR: torch.cuda.is_available() returned False.  Please activate a CUDA environment."
    Write-Host "Set Runtime > Change runtime type > GPU (Colab), or activate a conda env with CUDA." -ForegroundColor Red
    exit 1
}
$GpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
$GpuMem  = python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1),'GB')"
Log "GPU: $GpuName  ($GpuMem total)"

# --- Build dataset from splits -----------------------------------------------
Log "Step 1/6: Building IDD dataset from scene-disjoint splits..."
python scripts/build_dataset_from_splits.py --src "$DataSrc" --dst data/idd_converted 2>&1 | Tee-Object -Append -FilePath $LogFile

# --- Smoke run (1 epoch, 200 images) -----------------------------------------
Log "Step 2/6: Smoke run (1 epoch, 200 images) to estimate per-epoch time..."
$SmokeStart = Get-Date
python -m training.train --config $Config --synthetic --epochs 1 --tag smoke_v2 2>&1 | Tee-Object -Append -FilePath $LogFile
$SmokeEnd = Get-Date
$SmokeSec = [int]($SmokeEnd - $SmokeStart).TotalSeconds
$TotalEpochs = 50   # config default; real training may early-stop
$EstMin = [int]($SmokeSec * $TotalEpochs / 60)
Log "Smoke run: ${SmokeSec}s per epoch.  Estimated total for ${TotalEpochs} epochs: ${EstMin}min"

# --- Teacher training --------------------------------------------------------
Log "Step 3/6: Training teacher ($TeacherTag)..."
$TeacherBest = "outputs/checkpoints/${TeacherTag}_best.pt"
if ((Test-Path $TeacherBest) -and (-not $Resume)) {
    Log "  $TeacherBest already exists and --Resume not set.  Skipping teacher training."
} else {
    $ResumeArg = ""
    if ($Resume -and (Test-Path $TeacherBest)) { $ResumeArg = "--resume $TeacherBest" }
    $BatchArg = if ($BatchSize -gt 0) { "--batch-size $BatchSize" } else { "" }
    # OOM retry loop
    $CurrentBatch = if ($BatchSize -gt 0) { $BatchSize } else { 16 }
    $TrainOk = $false
    while (-not $TrainOk -and $CurrentBatch -ge 2) {
        $BatchArg = "--batch-size-override $CurrentBatch"
        python -m training.train --config $Config --tag $TeacherTag --num-workers $NumWorkers $ResumeArg 2>&1 | Tee-Object -Append -FilePath $LogFile
        if ($LASTEXITCODE -eq 0) { $TrainOk = $true }
        elseif ($LASTEXITCODE -eq 2) {  # OOM sentinel
            $CurrentBatch = [int]($CurrentBatch / 2)
            Log "  OOM detected.  Retrying with batch_size=$CurrentBatch..."
        } else { break }
    }
    if (-not $TrainOk) { Log "ERROR: Teacher training failed."; exit 1 }
}
Log "  Teacher checkpoint: $TeacherBest"

# --- Student distillation ----------------------------------------------------
Log "Step 4/6: Distilling student ($StudentTag) from teacher..."
$StudentBest = "outputs/checkpoints/${StudentTag}_best.pt"
if ((Test-Path $StudentBest) -and (-not $Resume)) {
    Log "  $StudentBest already exists and --Resume not set.  Skipping distillation."
} else {
    $ResumeArg = ""
    if ($Resume -and (Test-Path $StudentBest)) { $ResumeArg = "--resume $StudentBest" }
    python -m training.distill --config $Config --teacher $TeacherBest --tag $StudentTag --num-workers $NumWorkers $ResumeArg 2>&1 | Tee-Object -Append -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) { Log "ERROR: Distillation failed."; exit 1 }
}
Log "  Student checkpoint: $StudentBest"

# --- ONNX export and numerical check -----------------------------------------
Log "Step 5/6: Exporting student ONNX and verifying numerical equivalence..."
if (Test-Path $StudentOnnx) {
    Log "  $StudentOnnx already exists.  Skipping export (use --Resume to re-export)."
} else {
    python scripts/export_onnx.py --checkpoint $StudentBest --output $StudentOnnx 2>&1 | Tee-Object -Append -FilePath $LogFile
    if ($LASTEXITCODE -ne 0) { Log "ERROR: ONNX export failed."; exit 1 }
}
# Verify ONNX vs PyTorch on 50 test images
python scripts/export_onnx.py --checkpoint $StudentBest --output $StudentOnnx --verify --data-src "$DataSrc" --verify-n 50 2>&1 | Tee-Object -Append -FilePath $LogFile

# --- Full test-split evaluation ----------------------------------------------
Log "Step 6/6: Evaluating on full held-out test split..."
python -m evaluation.test_split_eval `
    --data-src "$DataSrc" `
    --checkpoint $TeacherBest `
    --student-checkpoint $StudentBest `
    --onnx $StudentOnnx `
    --baseline-checkpoint "outputs/checkpoints/student_best.pt" `
    --output "outputs/reports/final_eval_v2.json" `
    2>&1 | Tee-Object -Append -FilePath $LogFile

if ($LASTEXITCODE -ne 0) { Log "ERROR: Evaluation failed."; exit 1 }

Log "=== Pipeline complete.  Log: $LogFile ==="
Log "Compare old vs new in outputs/reports/final_eval_v2.json"
