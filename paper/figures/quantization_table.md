| dataset   | setting                     | method                   |   route_rate |   sensitivity |   accuracy |   floor_met |   lost_ok |
|:----------|:----------------------------|:-------------------------|-------------:|--------------:|-----------:|------------:|----------:|
| pathmnist | FP32                        | always_small             |         0    |         95.67 |      92.21 |      nan    |    nan    |
| pathmnist | FP32                        | always_large             |       100    |         97.11 |      92.99 |      nan    |    nan    |
| pathmnist | FP32                        | miss_risk ltt a=0.97     |         0    |         95.67 |      92.21 |      100    |    100    |
| pathmnist | FP32                        | msp ltt a=0.97           |         0    |         95.67 |      92.21 |      100    |    100    |
| pathmnist | FP32                        | miss_risk ltt a=0.99     |         0    |         95.67 |      92.21 |       33.33 |      0    |
| pathmnist | FP32                        | msp ltt a=0.99           |         0    |         95.67 |      92.21 |       33.33 |      0    |
| pathmnist | FP32                        | miss_risk plugin a=0.97  |         0    |         95.67 |      92.21 |      100    |    100    |
| pathmnist | FP32                        | msp plugin a=0.97        |         0    |         95.67 |      92.21 |      100    |    100    |
| pathmnist | FP32                        | miss_risk plugin a=0.99  |         0    |         95.67 |      92.21 |       33.33 |      0    |
| pathmnist | FP32                        | msp plugin a=0.99        |         0    |         95.67 |      92.21 |       33.33 |      0    |
| pathmnist | INT8 naive                  | always_small             |         0    |          0.11 |      43.71 |      nan    |    nan    |
| pathmnist | INT8 naive                  | always_large             |       100    |         96.86 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 naive                  | miss_risk ltt a=0.97     |        78.1  |         95.03 |      88.27 |       66.67 |    100    |
| pathmnist | INT8 naive                  | msp ltt a=0.97           |        79.28 |         94.94 |      89.55 |       66.67 |     66.67 |
| pathmnist | INT8 naive                  | miss_risk ltt a=0.99     |        87.39 |         96.73 |      91.48 |      100    |    100    |
| pathmnist | INT8 naive                  | msp ltt a=0.99           |        87.52 |         96.65 |      91.82 |      100    |    100    |
| pathmnist | INT8 naive                  | miss_risk plugin a=0.97  |        76    |         94.38 |      87.54 |       66.67 |     66.67 |
| pathmnist | INT8 naive                  | msp plugin a=0.97        |        77.39 |         94.27 |      89.05 |       66.67 |     66.67 |
| pathmnist | INT8 naive                  | miss_risk plugin a=0.99  |        83.16 |         96.24 |      90.07 |      100    |    100    |
| pathmnist | INT8 naive                  | msp plugin a=0.99        |        83.44 |         96.13 |      90.75 |       66.67 |     66.67 |
| pathmnist | INT8 naive, FP32 thresholds | always_small             |         0    |          0.11 |      43.71 |      nan    |    nan    |
| pathmnist | INT8 naive, FP32 thresholds | always_large             |       100    |         96.86 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk ltt a=0.97     |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp ltt a=0.97           |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk ltt a=0.99     |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp ltt a=0.99           |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk plugin a=0.97  |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp plugin a=0.97        |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk plugin a=0.99  |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp plugin a=0.99        |         0    |          0.11 |      43.71 |        0    |      0    |
| pathmnist | INT8 mixed                  | always_small             |         0    |         94.11 |      91.49 |      nan    |    nan    |
| pathmnist | INT8 mixed                  | always_large             |       100    |         96.86 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 mixed                  | miss_risk ltt a=0.97     |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed                  | msp ltt a=0.97           |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed                  | miss_risk ltt a=0.99     |         0.16 |         94.46 |      91.56 |        0    |      0    |
| pathmnist | INT8 mixed                  | msp ltt a=0.99           |         0.62 |         94.43 |      91.68 |        0    |      0    |
| pathmnist | INT8 mixed                  | miss_risk plugin a=0.97  |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed                  | msp plugin a=0.97        |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed                  | miss_risk plugin a=0.99  |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | INT8 mixed                  | msp plugin a=0.99        |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | always_small             |         0    |         94.11 |      91.49 |      nan    |    nan    |
| pathmnist | INT8 mixed, FP32 thresholds | always_large             |       100    |         96.86 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.97     |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | msp ltt a=0.97           |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.99     |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | msp ltt a=0.99           |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.97  |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | msp plugin a=0.97        |         0    |         94.11 |      91.49 |       66.67 |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.99  |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | msp plugin a=0.99        |         0    |         94.11 |      91.49 |        0    |      0    |
| pathmnist | FP32-int8naive agreement    | mobilenet_v3_small seed0 |       nan    |        nan    |      46.03 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | mobilenet_v3_small seed0 |       nan    |        nan    |      97.53 |      nan    |    nan    |
| pathmnist | FP32-int8naive agreement    | resnet18 seed0           |       nan    |        nan    |      99.22 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | resnet18 seed0           |       nan    |        nan    |      99.22 |      nan    |    nan    |
| pathmnist | FP32-int8naive agreement    | mobilenet_v3_small seed1 |       nan    |        nan    |      48.87 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | mobilenet_v3_small seed1 |       nan    |        nan    |      98.02 |      nan    |    nan    |
| pathmnist | FP32-int8naive agreement    | resnet18 seed1           |       nan    |        nan    |      98.82 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | resnet18 seed1           |       nan    |        nan    |      98.82 |      nan    |    nan    |
| pathmnist | FP32-int8naive agreement    | mobilenet_v3_small seed2 |       nan    |        nan    |      41.64 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | mobilenet_v3_small seed2 |       nan    |        nan    |      96.53 |      nan    |    nan    |
| pathmnist | FP32-int8naive agreement    | resnet18 seed2           |       nan    |        nan    |      99.14 |      nan    |    nan    |
| pathmnist | FP32-int8 agreement         | resnet18 seed2           |       nan    |        nan    |      99.14 |      nan    |    nan    |
| ham10000  | FP32                        | always_small             |         0    |         63.92 |      81.53 |      nan    |    nan    |
| ham10000  | FP32                        | always_large             |       100    |         64.76 |      82.97 |      nan    |    nan    |
| ham10000  | FP32                        | miss_risk ltt a=0.97     |        34.61 |         70.97 |      83.37 |      100    |    100    |
| ham10000  | FP32                        | msp ltt a=0.97           |        44.54 |         65.23 |      83.6  |      100    |    100    |
| ham10000  | FP32                        | miss_risk ltt a=0.99     |       100    |         64.76 |      82.97 |      100    |    100    |
| ham10000  | FP32                        | msp ltt a=0.99           |       100    |         64.76 |      82.97 |      100    |    100    |
| ham10000  | FP32                        | miss_risk plugin a=0.97  |         0.58 |         64.4  |      81.43 |       66.67 |      0    |
| ham10000  | FP32                        | msp plugin a=0.97        |        10.85 |         65.47 |      82.23 |      100    |     33.33 |
| ham10000  | FP32                        | miss_risk plugin a=0.99  |         1.05 |         64.99 |      81.53 |       66.67 |      0    |
| ham10000  | FP32                        | msp plugin a=0.99        |        15.14 |         65.23 |      82.09 |      100    |      0    |
| ham10000  | INT8 naive                  | always_small             |         0    |          0.12 |      64.71 |      nan    |    nan    |
| ham10000  | INT8 naive                  | always_large             |       100    |         65.35 |      82.93 |      nan    |    nan    |
| ham10000  | INT8 naive                  | miss_risk ltt a=0.97     |        98.84 |         64.76 |      82.88 |      100    |    100    |
| ham10000  | INT8 naive                  | msp ltt a=0.97           |        99.21 |         64.99 |      82.88 |      100    |    100    |
| ham10000  | INT8 naive                  | miss_risk ltt a=0.99     |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 naive                  | msp ltt a=0.99           |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 naive                  | miss_risk plugin a=0.97  |        96.09 |         62.84 |      82.34 |       33.33 |     33.33 |
| ham10000  | INT8 naive                  | msp plugin a=0.97        |        96.39 |         62.72 |      82.6  |       33.33 |     33.33 |
| ham10000  | INT8 naive                  | miss_risk plugin a=0.99  |        98.84 |         64.76 |      82.88 |       33.33 |     33.33 |
| ham10000  | INT8 naive                  | msp plugin a=0.99        |        99.21 |         64.99 |      82.88 |       66.67 |     66.67 |
| ham10000  | INT8 naive, FP32 thresholds | always_small             |         0    |          0.12 |      64.71 |      nan    |    nan    |
| ham10000  | INT8 naive, FP32 thresholds | always_large             |       100    |         65.35 |      82.93 |      nan    |    nan    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk ltt a=0.97     |        36.43 |         18.76 |      72.23 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | msp ltt a=0.97           |        39.04 |         19    |      72.58 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk ltt a=0.99     |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | msp ltt a=0.99           |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk plugin a=0.97  |         0.07 |          0.12 |      64.73 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | msp plugin a=0.97        |         5.45 |          3.7  |      65.9  |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk plugin a=0.99  |         0.16 |          0.36 |      64.78 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | msp plugin a=0.99        |        12.21 |          7.17 |      66.97 |        0    |      0    |
| ham10000  | INT8 mixed                  | always_small             |         0    |         65.23 |      80.41 |      nan    |    nan    |
| ham10000  | INT8 mixed                  | always_large             |       100    |         65.35 |      82.93 |      nan    |    nan    |
| ham10000  | INT8 mixed                  | miss_risk ltt a=0.97     |        37.41 |         73.36 |      83.83 |      100    |    100    |
| ham10000  | INT8 mixed                  | msp ltt a=0.97           |        47.94 |         67.03 |      83.55 |      100    |    100    |
| ham10000  | INT8 mixed                  | miss_risk ltt a=0.99     |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 mixed                  | msp ltt a=0.99           |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 mixed                  | miss_risk plugin a=0.97  |         0.14 |         65.35 |      80.43 |      100    |      0    |
| ham10000  | INT8 mixed                  | msp plugin a=0.97        |         3.14 |         66.07 |      81.3  |      100    |      0    |
| ham10000  | INT8 mixed                  | miss_risk plugin a=0.99  |         0.51 |         65.47 |      80.46 |       66.67 |      0    |
| ham10000  | INT8 mixed                  | msp plugin a=0.99        |         7.99 |         66.91 |      81.57 |      100    |      0    |
| ham10000  | INT8 mixed, FP32 thresholds | always_small             |         0    |         65.23 |      80.41 |      nan    |    nan    |
| ham10000  | INT8 mixed, FP32 thresholds | always_large             |       100    |         65.35 |      82.93 |      nan    |    nan    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.97     |        41.81 |         72.88 |      83.53 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | msp ltt a=0.97           |        52.48 |         66.67 |      83.48 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.99     |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | msp ltt a=0.99           |       100    |         65.35 |      82.93 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.97  |         0.56 |         65.59 |      80.46 |      100    |      0    |
| ham10000  | INT8 mixed, FP32 thresholds | msp plugin a=0.97        |        12.79 |         66.79 |      81.67 |      100    |     33.33 |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.99  |         1.09 |         66.19 |      80.53 |      100    |      0    |
| ham10000  | INT8 mixed, FP32 thresholds | msp plugin a=0.99        |        18.91 |         66.19 |      81.41 |      100    |     33.33 |
| ham10000  | FP32-int8naive agreement    | mobilenet_v3_small seed0 |       nan    |        nan    |      66.74 |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | mobilenet_v3_small seed0 |       nan    |        nan    |      88.05 |      nan    |    nan    |
| ham10000  | FP32-int8naive agreement    | resnet18 seed0           |       nan    |        nan    |      98.6  |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | resnet18 seed0           |       nan    |        nan    |      98.6  |      nan    |    nan    |
| ham10000  | FP32-int8naive agreement    | mobilenet_v3_small seed1 |       nan    |        nan    |      70.16 |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | mobilenet_v3_small seed1 |       nan    |        nan    |      89.24 |      nan    |    nan    |
| ham10000  | FP32-int8naive agreement    | resnet18 seed1           |       nan    |        nan    |      98.18 |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | resnet18 seed1           |       nan    |        nan    |      98.18 |      nan    |    nan    |
| ham10000  | FP32-int8naive agreement    | mobilenet_v3_small seed2 |       nan    |        nan    |      67.3  |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | mobilenet_v3_small seed2 |       nan    |        nan    |      90.08 |      nan    |    nan    |
| ham10000  | FP32-int8naive agreement    | resnet18 seed2           |       nan    |        nan    |      98.67 |      nan    |    nan    |
| ham10000  | FP32-int8 agreement         | resnet18 seed2           |       nan    |        nan    |      98.67 |      nan    |    nan    |