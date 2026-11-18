# x86 reference benchmarks on this PC: every cascade_*.json in a deploy folder x thread counts.
#   powershell -ExecutionPolicy Bypass -File scripts\bench_pc.ps1 -Deploy deploy\pathmnist_128_seed0
param(
    [string]$Deploy = "deploy\pathmnist_128_seed0",
    [string]$Threads = "1,2,4",
    [string]$Tag = "ryzen7700"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
$out = "results\bench\$Tag"
foreach ($cfg in Get-ChildItem "$Deploy\cascade_*.json") {
    $precision = ($cfg.Name -split "_")[1]  # cascade_<fp32|int8|int8naive>_...
    foreach ($t in $Threads.Split(",")) {
        & $py edge_bench\bench.py --deploy $Deploy --precision $precision --config $cfg.Name --threads ([int]$t) --tag $Tag --out $out
        if ($LASTEXITCODE -ne 0) { throw "bench failed: $($cfg.Name) t=$t" }
        Start-Sleep -Seconds 5
    }
}
