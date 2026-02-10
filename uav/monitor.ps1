# Monitor experiment: show last N lines (ASCII/English progress). Refresh every 10s.
# Usage: powershell -ExecutionPolicy Bypass -File monitor.ps1
# Full run ETA: ~30-50 min on AMD GPU (2 models, 16 tasks, all questions)
$base = Join-Path $env:USERPROFILE ".cursor\projects"
$terminals = Get-ChildItem $base -Recurse -Filter "*.txt" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "terminals\\d+\.txt$" }
$latest = $terminals | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$tail = 28
Write-Host "Monitor (last $tail lines, 10s). Full run ETA: ~30-50 min GPU. Ctrl+C to stop."
Write-Host ""
while ($true) {
    Clear-Host
    Write-Host "=== MM-UAVBench Monitor $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
    if ($latest -and (Test-Path $latest.FullName)) {
        $lines = Get-Content $latest.FullName -Tail $tail -Encoding UTF8 -ErrorAction SilentlyContinue
        $etaLine = $lines | Where-Object { $_ -match "ETA\s+([\d.]+)\s*min" } | Select-Object -Last 1
        if ($etaLine -match "ETA\s+([\d.]+)\s*min") {
            Write-Host ">>> Estimated remaining: $($Matches[1]) min <<<" -ForegroundColor Yellow
            Write-Host ""
        }
        $lines | ForEach-Object { Write-Host $_ }
    } else {
        $terminals = Get-ChildItem $base -Recurse -Filter "*.txt" -ErrorAction SilentlyContinue | Where-Object { $_.FullName -match "terminals\\d+\.txt$" }
        $latest = $terminals | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            $lines = Get-Content $latest.FullName -Tail $tail -Encoding UTF8 -ErrorAction SilentlyContinue
            $etaLine = $lines | Where-Object { $_ -match "ETA\s+([\d.]+)\s*min" } | Select-Object -Last 1
            if ($etaLine -match "ETA\s+([\d.]+)\s*min") {
                Write-Host ">>> Estimated remaining: $($Matches[1]) min <<<" -ForegroundColor Yellow
                Write-Host ""
            }
            $lines | ForEach-Object { Write-Host $_ }
        } else { Write-Host "No terminal log. Run experiment first: run.bat" }
    }
    Start-Sleep -Seconds 10
}
