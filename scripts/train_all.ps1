# Train Stage 1 + Stage 2 for every seed, then analyse FP32 results.
#   powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1 -Dataset pathmnist -Seeds "0,1,2"
# (Seeds is a string: with -File, PowerShell does not parse 0,1,2 as an array and would pass 12.)
param(
    [string]$Dataset = "pathmnist",
    [string]$Seeds = "0,1,2",
    [int]$Size = 128
)
$SeedList = $Seeds.Split(",") | ForEach-Object { [int]$_.Trim() }
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
foreach ($seed in $SeedList) {
    foreach ($arch in "mobilenet_v3_small", "resnet18") {
        $done = "runs\${Dataset}_${Size}\$arch\seed$seed\logits_test.npz"
        if (Test-Path $done) { Write-Host "skip $arch seed $seed (done)"; continue }
        & $py -m cascade.train --dataset $Dataset --size $Size --arch $arch --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "training failed: $arch seed $seed" }
    }
}
& $py -m cascade.analyze --dataset $Dataset --size $Size
