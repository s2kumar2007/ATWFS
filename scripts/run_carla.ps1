[CmdletBinding()]
param([switch]$Smoke,[string]$Scenarios='all',[int]$Runs=10,[string]$CarlaExe,[string]$Town='Town01',[ValidateSet('low','epic')][string]$Quality='low')
$ErrorActionPreference='Stop'
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Warning 'nvidia-smi unavailable; GPU check cannot run.' } else { nvidia-smi }
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
if ($CarlaExe) { if (-not (Test-Path -LiteralPath $CarlaExe)) { throw "CARLA executable missing: $CarlaExe" }; Start-Process -FilePath $CarlaExe -ArgumentList @("$Town", "-quality-level=$Quality") -WindowStyle Hidden }
python -m sim.run_carla --smoke
if ($LASTEXITCODE -ne 0) { throw 'CARLA smoke test failed; see outputs/sim_carla run_log.txt' }
if (-not $Smoke) { python -m sim.run_carla --scenario $Scenarios --runs $Runs }
if ($LASTEXITCODE -ne 0) { throw 'CARLA run failed; see outputs/sim_carla run_log.txt' }
