param(
    [ValidateSet('check','prepare','train','calibrate','attack','final','all')]
    [string]$Stage = 'check',
    [string]$Python = 'python',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONUNBUFFERED = '1'
    $runArgs = @('-m', 'scripts.local_gpu', $Stage)
    if ($DryRun) { $runArgs += '--dry-run' }
    & $Python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE. Check results/local_logs." }
} finally {
    Pop-Location
}
