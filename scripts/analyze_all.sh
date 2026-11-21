#!/usr/bin/env sh
# Every analysis variant for both datasets, then device-benchmark configs. Run from the project root (Git Bash):
#   sh scripts/analyze_all.sh
set -e
PY=.venv/Scripts/python.exe
for ds in pathmnist ham10000; do
  for calib in val target; do
    $PY -m cascade.analyze --dataset $ds --calib $calib --n-boot 0 | tail -1
    for v in int8 int8naive; do
      $PY -m cascade.analyze --dataset $ds --variant $v --calib $calib --n-boot 0 | tail -1
      $PY -m cascade.analyze --dataset $ds --variant $v --select-variant fp32 --calib $calib --n-boot 0 | tail -1
    done
  done
done
# Bootstrap CIs only for the headline FP32 analyses
for ds in pathmnist ham10000; do
  $PY -m cascade.analyze --dataset $ds --calib val --n-boot 1000 | tail -1
done

# Class-balanced HAM10000 robustness check (FP32 only; train with scripts/train_all.ps1 -Dataset ham10000bal)
if [ -d runs/ham10000bal_128 ]; then
  $PY -m cascade.analyze --dataset ham10000bal --calib val --n-boot 1000 | tail -1
  $PY -m cascade.analyze --dataset ham10000bal --calib target --n-boot 0 | tail -1
fi

# Operating points benchmarked on devices (seed 0). PathMNIST: target-site calibration, paired LTT.
for v in fp32 int8; do
  for sig in msp miss_risk; do
    for a in 0.97 0.99; do
      $PY -m cascade.deploy config --dataset pathmnist --seed 0 --variant $v --signal $sig --alpha $a --rule ltt --calib target > /dev/null
    done
  done
done
$PY -m cascade.deploy config --dataset pathmnist --seed 0 --variant int8naive --signal miss_risk --alpha 0.97 --rule ltt --calib target > /dev/null
# HAM10000: source-validation calibration (the target split is too small to certify a floor), LTT alpha = 0.97.
for v in fp32 int8; do
  for sig in msp miss_risk; do
    $PY -m cascade.deploy config --dataset ham10000 --seed 0 --variant $v --signal $sig --alpha 0.97 --rule ltt --calib val > /dev/null
  done
done
ls deploy/pathmnist_128_seed0/cascade_*.json deploy/ham10000_128_seed0/cascade_*.json
