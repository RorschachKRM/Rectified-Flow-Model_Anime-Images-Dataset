param(
    [int]$TrainingPid = 90612
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$stdoutLog = Join-Path $projectRoot "outputs\v1_2\local_training_stdout.log"
$stderrLog = Join-Path $projectRoot "outputs\v1_2\local_training_stderr.log"
$latestCheckpoint = Join-Path $projectRoot "outputs\v1_2\checkpoints\latest.pt"

while ($true) {
    $trainingProcess = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    Clear-Host
    Write-Host "KRM Rectified Flow V1.2 - Local Training Monitor" -ForegroundColor Cyan
    Write-Host "Process ID: $TrainingPid"
    Write-Host "Status: $(if ($trainingProcess) { 'RUNNING' } else { 'STOPPED' })"
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    Write-Host "`nGPU" -ForegroundColor Yellow
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader

    if (Test-Path -LiteralPath $latestCheckpoint) {
        $checkpoint = Get-Item -LiteralPath $latestCheckpoint
        Write-Host "`nLatest checkpoint: $($checkpoint.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) / $([math]::Round($checkpoint.Length / 1MB, 1)) MiB" -ForegroundColor Yellow
    }

    Write-Host "`nCompleted epochs" -ForegroundColor Yellow
    if (Test-Path -LiteralPath $stdoutLog) {
        Get-Content -LiteralPath $stdoutLog -Encoding Default |
            Where-Object { $_ -match '^Epoch ' } |
            Select-Object -Last 6
    }

    Write-Host "`nCurrent batch" -ForegroundColor Yellow
    if (Test-Path -LiteralPath $stderrLog) {
        Get-Content -LiteralPath $stderrLog -Tail 40 -Encoding Default -ErrorAction SilentlyContinue |
            Where-Object { $_ -match 'Train:|Validation:' } |
            Select-Object -Last 1
    }

    if (-not $trainingProcess) {
        Write-Host "`nTraining process has finished. Press Enter to close." -ForegroundColor Green
        Read-Host
        break
    }
    Start-Sleep -Seconds 2
}
