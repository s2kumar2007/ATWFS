[CmdletBinding()]
param([switch]$Smoke,[string]$Scenarios='all',[int]$Runs=10,[string]$CarlaExe,[string]$Town='Town01',[ValidateSet('low','epic')][string]$Quality='low',[string]$OnnxPath='outputs/checkpoints/student_v2.onnx',[switch]$Mock)
$ErrorActionPreference='Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$runRoot=Join-Path (Get-Location) ('outputs/sim_carla/launcher_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
Start-Transcript -Path (Join-Path $runRoot 'launcher_log.txt') | Out-Null
try {
if ($Runs -lt 1) { throw 'Runs must be positive.' }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Warning 'nvidia-smi unavailable; GPU check cannot run.' } else { nvidia-smi }
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('WARNING: CPU inference' if not torch.cuda.is_available() else torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch cannot import. Install project requirements.' }
if (-not $Mock) {
    python -c "import carla; print('CARLA Python API imported')"
    if ($LASTEXITCODE -ne 0) { throw 'Install the CARLA Python wheel matching the server release.' }
    if (-not (Test-Path -LiteralPath $OnnxPath -PathType Leaf)) { throw "ONNX model missing: $OnnxPath" }
}
$qualityName=if ($Quality -eq 'low') { 'Low' } else { 'Epic' }
if ($CarlaExe -and -not $Mock) { if (-not (Test-Path -LiteralPath $CarlaExe)) { throw "CARLA executable missing: $CarlaExe" }; Start-Process -FilePath $CarlaExe -ArgumentList @("$Town", "-quality-level=$qualityName") -WindowStyle Hidden }
$commonArgs=@('-m','sim.run_carla','--town',$Town)
if ($Mock) { $commonArgs+='--mock' } else { $commonArgs+=@('--onnx-path',$OnnxPath) }
if ($CarlaExe -and -not $Mock) {
    $ready=$false
    $deadline=(Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        $tcp=[System.Net.Sockets.TcpClient]::new()
        try { $task=$tcp.ConnectAsync('localhost',2000); if ($task.Wait(500) -and $tcp.Connected) { $ready=$true; break } } catch { } finally { $tcp.Dispose() }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) { throw 'CARLA did not open port 2000. Check server logs/firewall.' }
}
python @commonArgs --smoke --output-dir (Join-Path $runRoot 'smoke')
if ($LASTEXITCODE -ne 0) { throw 'CARLA smoke test failed; see outputs/sim_carla run_log.txt' }
if (-not $Smoke) { python @commonArgs --scenario $Scenarios --runs $Runs --output-dir (Join-Path $runRoot 'experiment') }
if ($LASTEXITCODE -ne 0) { throw 'CARLA run failed; see outputs/sim_carla run_log.txt' }

Write-Host "Send the complete folder: $runRoot"
} finally { Stop-Transcript | Out-Null; Pop-Location }
