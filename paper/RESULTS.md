# Results so far (PC + Android complete; Oracle A1 pending)

Status: 8 Nov 2026. Every number below comes from the files in `results/` and `paper/figures/`.
Setup: 128 × 128 px. MobileNetV3-Small (Stage 1) → ResNet-18 (Stage 2), ImageNet-pretrained, 3 seeds each.
"Target" means thresholds are selected on 30% of the deployment-centre data and evaluated on the other 70%
(10 random splits per seed). "Source" means they are selected on the validation split.

## 1. Baselines (test, mean of 3 seeds)

| Dataset | Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| PathMNIST | MobileNetV3-Small | 92.21 | 95.67 | 98.83 |
| PathMNIST | ResNet-18 | 92.99 | 97.11 | 99.31 |
| PathMNIST | Oracle cascade (escalates 3.1%) | 95.31 | 98.13 | 99.55 |
| HAM10000 | MobileNetV3-Small | 81.53 | 63.92 | 93.52 |
| HAM10000 | ResNet-18 | 82.97 | 64.76 | 94.44 |
| HAM10000 | Oracle cascade (escalates 7.4%) | 88.91 | 75.1 | 96.44 |

Validation accuracy on PathMNIST is about 99.8% for both models; the cross-centre test split is where they differ.

## 2. Key findings

**F1 — Source-site thresholds fail under domain shift (H5 confirmed).**
- Calibrated on PathMNIST's validation split, every signal escalates **0%** of images at α ≤ 0.99, because both models are saturated there.
- On the cross-centre test set, the α = 0.99 floor then holds in only 1 of 3 seeds.
- At α = 1.00, the plug-in rule escalates 1.5% and misses the floor in **every** seed.
- Thresholds have to be calibrated at the deployment site.

**F2 — Miss-risk escalates far fewer images at the same sensitivity (H3 confirmed).**

PathMNIST, target calibration, paired LTT (95% guarantee):

| Floor | MSP | Entropy | Miss-risk |
|---|---|---|---|
| α = 0.97 | 4.8% escalated | 5.1% | **1.2%** |
| α = 0.99 | 23.0% | 23.2% | **16.3%** |
| α = 1.00 (plug-in) | 14.7% | 15.1% | **5.9%** |

Fixed escalation budgets (full test set, 3 seeds; `paper/figures/budget_table.md`):

| Budget | PathMNIST, MSP → miss-risk sensitivity | HAM10000, MSP → miss-risk sensitivity |
|---|---|---|
| 2% | 96.05 → **97.22** (above ResNet-18's 97.11) | 64.5 → 66.0 |
| 10% | 96.89 → 97.22 | 65.8 → **72.4** |
| 20% | 97.03 → 97.13 | 66.8 → **76.0** (ResNet-18 alone: 64.8) |

The accuracy cost on HAM10000 is about 1 point at a 20% budget.

AUROC for ranking Stage-1 *missed positives*:

| Dataset | MSP | Miss-risk |
|---|---|---|
| PathMNIST | 0.933 | **0.965** |
| HAM10000 | 0.605 | **0.888** |

**F3 — Temperature scaling barely changes routing (H2 confirmed).**
- Stage-1 test ECE:

  | Dataset | Before TS | After TS |
  |---|---|---|
  | PathMNIST | 5.5% | 4.7% |
  | HAM10000 | **14.0%** | **2.4%** |

- Escalation rates for TS-MSP and MSP differ by less than 1 point in most settings and by at most 3.4 points (HAM10000, source LTT α = 0.97: 44.5% vs. 47.9%). Error-ranking AUROC is identical (PathMNIST 0.853 vs. 0.852).
- Calibration fixes confidence values but not their ranking, and re-selecting the threshold absorbs the rest.

**F4 — Finite-sample guarantees need enough local positives.**
- The paired LTT (lost-detection rate ≤ (1−α)·S_L) certifies α = 0.97 and 0.99 on PathMNIST, which has about 370 positives in the 30% calibration split. Floors were met in 100% of held-out evaluations.
- On HAM10000, about 84 positives in the target calibration split are too few to certify any floor, so LTT escalates everything. About 280 positives (the source validation split) are enough for α = 0.97: miss-risk escalates 34.6% and reaches **71.0%** sensitivity, vs. 64.8% for ResNet-18 and 44.5% escalation for MSP.
- α = 1.00 can never be certified with finite data, so LTT correctly escalates everything.

**F5 — Naive INT8 post-training quantisation breaks Stage 1, and silently breaks the cascade (RQ4).**
- Full static INT8 MobileNetV3-Small agrees with FP32 on only 42–49% (PathMNIST) and 67–70% (HAM10000) of predictions, and its sensitivity collapses to about 0%.
- ResNet-18 INT8 is fine (≈98.5% agreement).
- Per-block analysis shows the cause: quantising just the first conv, block 1, or the hard-swish activations is enough on its own.
- **Mixed precision** (those kept in FP32) restores agreement to 96.5–98.0% (PathMNIST) and 88–90% (HAM10000), with accuracy within about 1 point of FP32.

PathMNIST, target calibration, LTT, α = 0.97, miss-risk:

| Stage-1 model | Thresholds | Escalated | Sensitivity | Floor met |
|---|---|---|---|---|
| FP32 | own | 1.2% | 96.6 | 100% |
| INT8 naive | FP32's | 4.4% | **6.7** | **0%** |
| INT8 naive | re-tuned | 78.4% | 95.2 | 100% (efficiency lost) |
| INT8 mixed | FP32's | 1.6% | 95.6 | 100% (lost-detection bound held in only 83%) |
| INT8 mixed | re-tuned | 3.1% | 96.1 | 100% |

**F6 — Measured latency savings fall well short of FLOP predictions (H4 confirmed; x86 reference so far).**
- FLOPs: MobileNetV3-Small 37.7 M vs. ResNet-18 1,184 M, a ratio of **31.4×**.
- Measured Stage-2/Stage-1 latency ratio on the Ryzen 7 7700 (batch 1, ONNX Runtime):

  | Precision | 1 thread | 4 threads |
  |---|---|---|
  | FP32 | 12–13× | 4.7–5.4× |
  | INT8 | **3.6×** | **1.6×** |

  INT8 makes ResNet-18 3.4× faster (8.5 → 2.5 ms, mean of the four configurations), but mixed-INT8 MobileNet is no faster than FP32 (0.68 ms in both).
- Measured cascade latency matches the stage-latency model t₁ + r·t₂ almost exactly (`paper/figures/speedup_vs_escalation.png`). FLOP counts over-predict the speed-up by up to 2.2× in FP32 and 5× in INT8.

| PathMNIST, 1 thread | Escalated | Cascade (ms) | Speed-up measured | Speed-up predicted by FLOPs |
|---|---|---|---|---|
| FP32 miss-risk α = 0.97 | 0.4% | 0.71 | **12.5×** | 27.9× |
| FP32 miss-risk α = 0.99 | 12.2% | 1.74 | 4.8× | 6.5× |
| INT8 miss-risk α = 0.97 | 2.4% | 0.71 | 3.6× | 17.9× |
| INT8 msp α = 0.99 | 18.8% | 1.12 | 2.2× | 4.6× |
| INT8 *naive* miss-risk α = 0.97 | 90.6% | 3.11 | **0.83× (slower than always-large)** | 1.07× |

- Practical consequence: an FP32 cascade (1.74 ms at α = 0.99) beats INT8 ResNet-18 on its own (2.53 ms) at matching sensitivity. With 4 threads and INT8, the cascade advantage almost disappears (1.2–1.6×).
- Escalation rates measured on the device match the offline analysis to within 1.7 percentage points (500-image stratified subset).

**F6b — On a phone the gap is even larger, and INT8 behaves differently from x86.**
- Device: Oppo F19 (Snapdragon 662). Gold cluster = 4× Kryo 260 Gold (A73-class); Silver cluster = 4× Kryo 260 Silver (A53-class).
- Software: Android 13 / ColorOS 13; ONNX Runtime 1.31.1 in Termux + proot Ubuntu; plugged in during the run.
- Stability: 1-thread Stage-1 latency stayed at 14.5–14.8 ms across all 1.5 hours, so there was no visible throttling. The temperature sensor read a constant 37 °C, so it cannot be used as evidence either way.

| PathMNIST, 1 thread, Gold | Escalated | Cascade (ms) | Speed-up measured | Speed-up predicted by FLOPs |
|---|---|---|---|---|
| FP32 miss-risk α = 0.97 | 0.4% | 15.3 | **8.9×** | 27.9× |
| FP32 miss-risk α = 0.99 | 12.2% | 31.3 | 4.3× | 6.5× |
| INT8 miss-risk α = 0.97 | 2.4% | 9.8 | 6.6× | 17.9× |
| INT8 msp α = 0.99 | 18.8% | 20.4 | 3.2× | 4.5× |
| INT8 *naive* miss-risk α = 0.97 | 90.6% | 66.6 | **0.97×** | 1.07× |

Measured Stage-2/Stage-1 latency ratio (1 thread), against a FLOP ratio of 31.4×:

| Precision | Ryzen 7700 | Kryo Gold | Kryo Silver |
|---|---|---|---|
| FP32 | 12–13× | 9.2× | 8.9× |
| INT8 | 3.6× | 8.0× | 8.4× |

- On ARM, mixed-INT8 MobileNet *is* faster than FP32 (Gold: 14.6 → 8.1 ms, 1.8×); on the Ryzen it gave no gain. ResNet-18 INT8 gains 2.1× (135 → 64.5 ms) vs. 3.3× on the Ryzen.
- So the speed-up an INT8 cascade can reach depends on the platform: on the phone it stays near FP32 levels, while on x86 it is cut to under 4×.
- The Silver cores are about 2.3× slower in absolute terms but give the same relative speed-ups (FP32 α = 0.97: 8.5×).
- At 1 thread, the stage-latency model t₁ + r·t₂ matches measured cascade latency to within 2% on both clusters (Gold ≤1.5%, Silver ≤1.7%).
- Practical consequence: on the phone, an FP32 cascade at α = 0.99 (31 ms) is twice as fast as INT8 ResNet-18 alone (64.5 ms). The INT8 cascade at α = 0.99 is 16.7 ms.

**F7 — Thread settings matter more than quantisation for a two-model cascade on a 4-core phone cluster.**
- Setup: both ONNX Runtime sessions run on the same 4 Gold cores.
- **4 threads, default settings (spin-waiting on).** Each stage is fastest on its own (FP32: 7.2 / 44.6 ms). Measured cascade latency is then **+115% on average (up to +179%)** above t₁ + r·t₂, and the cascade is often slower than at 1 thread.
  - Cause: the idle session's worker threads keep spinning and compete with the active session.
  - Per-image traces show routed images take ~120 ms instead of the predicted ~52 ms, and accepted images straight after a routed one are 2–4× slower.
  - The Ryzen (16 hardware threads) shows no such effect (+3% at 4 threads).
- **4 threads, spinning off** (`session.intra_op.allow_spinning=0`). Contention disappears (model error +2% on average), but each stage becomes slower (FP32 Stage 1: 23.7 ms, worse than 1 thread). This is likely made worse by proot's ptrace-based system-call interception, which makes thread sleep/wake expensive.
- **2 threads per session** (2 sessions × 2 threads = 4 cores) gives the fastest PathMNIST cascade in 8 of 9 configurations and matches the model to within 4%. Example: FP32 miss-risk α = 0.97 is 15.3 / 10.0 / 8.0 / 25.0 ms at 1 / 2 / 4 / 4-no-spin threads; α = 0.99 is 31.3 / 18.9 / 28.1 / 32.4 ms.
- Practical rule: split the cores between the resident models instead of giving every session all of them.

| PathMNIST cascade (ms), Gold | 1 thread | 2 threads | 4 threads | 4 threads, no spin |
|---|---|---|---|---|
| FP32 miss-risk α = 0.97 | 15.3 | 10.0 | **8.0** | 25.0 |
| FP32 miss-risk α = 0.99 | 31.3 | **18.9** | 28.1 | 32.4 |
| FP32 msp α = 0.99 | 38.6 | **22.6** | 35.1 | 31.8 |
| INT8 miss-risk α = 0.97 | 9.8 | **7.7** | 9.2 | 17.2 |
| INT8 miss-risk α = 0.99 | 16.7 | **11.2** | 20.9 | 21.3 |
| INT8 msp α = 0.99 | 20.4 | **13.1** | 21.2 | 23.9 |

**F8 — Robustness: class-balanced HAM10000 training (stronger baselines; same data, split and architectures).**
- Training change: inverse-frequency class weights in the loss, and the checkpoint is chosen by balanced validation accuracy. Results are in `results/ham10000bal_128[_target]`.
- On-device latency is unchanged, so no device re-runs are needed.

| HAM10000, test (3 seeds) | Accuracy | Sensitivity | Specificity | Macro recall |
|---|---|---|---|---|
| MobileNetV3-Small, standard → balanced | 81.5 → 78.9 | 63.9 → **73.2** | 93.5 → 88.7 | 60.9 → 66.7 |
| ResNet-18, standard → balanced | 83.0 → 78.1 | 64.8 → **77.3** | 94.4 → 86.9 | 66.5 → 69.9 |

**The miss-risk advantage holds, but is smaller.** Class weighting already pushes the models towards the positive classes, which is part of what miss-risk does.

| Balanced models | MSP | Miss-risk |
|---|---|---|
| AUROC, missed positives (standard models: 0.605 / 0.888) | 0.720 | **0.936** |
| Sensitivity at a 10% escalation budget | 78.1 | **82.0** |
| Sensitivity at a 20% escalation budget (ResNet-18 alone: 77.3) | 79.5 | **84.0** |
| LTT, α = 0.97, source calibration: escalated / sensitivity | 59.3% / 78.5 | **47.0% / 81.6** |
| LTT, α = 0.95, **target** calibration: escalated / sensitivity | 50.0% / 78.6 | **38.4% / 83.4** |

- **Cost:** miss-risk trades specificity for sensitivity. At a 20% budget accuracy is 77.5 vs. 80.9 for MSP; it was 82.8 vs. 84.0 with the standard models.
- **Target calibration:** α = 0.95 can now be certified at the target site, because a higher S_L allows a larger lost-detection budget. α = 0.97 still cannot (about 84 local positives). F4 is unchanged.
- **Temperature scaling:** ECE drops from 10.9% to 2.2%, and routing again barely changes. F3 holds.

## 3. Still to do

1. Oracle Ampere A1: still waiting for capacity. Run `edge_bench/run_matrix.sh` there (README), then rerun `python -m cascade.figures`. Android (Gold + Silver) is done.
2. Writing: results, discussion and limitations. Verify the four `<!-- VERIFY -->` references.
