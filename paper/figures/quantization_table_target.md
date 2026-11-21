| dataset   | setting                     | method                   |   route_rate |   sensitivity |   accuracy |   floor_met |   lost_ok |
|:----------|:----------------------------|:-------------------------|-------------:|--------------:|-----------:|------------:|----------:|
| pathmnist | FP32                        | always_small             |         0    |         95.66 |      92.21 |      nan    |    nan    |
| pathmnist | FP32                        | always_large             |       100    |         97.08 |      92.99 |      nan    |    nan    |
| pathmnist | FP32                        | miss_risk ltt a=0.97     |         1.17 |         96.55 |      92.44 |      100    |    100    |
| pathmnist | FP32                        | msp ltt a=0.97           |         4.83 |         96.26 |      92.89 |      100    |    100    |
| pathmnist | FP32                        | miss_risk ltt a=0.99     |        16.32 |         97.29 |      93.06 |      100    |     93.33 |
| pathmnist | FP32                        | msp ltt a=0.99           |        23.03 |         97.11 |      93.23 |      100    |     93.33 |
| pathmnist | FP32                        | miss_risk plugin a=0.97  |         0.01 |         95.69 |      92.21 |      100    |     80    |
| pathmnist | FP32                        | msp plugin a=0.97        |         0.11 |         95.66 |      92.23 |      100    |     80    |
| pathmnist | FP32                        | miss_risk plugin a=0.99  |         0.91 |         96.32 |      92.37 |       63.33 |     10    |
| pathmnist | FP32                        | msp plugin a=0.99        |         4.37 |         96.23 |      92.8  |       60    |     16.67 |
| pathmnist | INT8 naive                  | always_small             |         0    |          0.09 |      43.73 |      nan    |    nan    |
| pathmnist | INT8 naive                  | always_large             |       100    |         96.84 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 naive                  | miss_risk ltt a=0.97     |        78.4  |         95.23 |      88.62 |      100    |    100    |
| pathmnist | INT8 naive                  | msp ltt a=0.97           |        79.5  |         95.27 |      90.03 |       90    |     90    |
| pathmnist | INT8 naive                  | miss_risk ltt a=0.99     |        87.9  |         96.57 |      91.24 |       93.33 |     93.33 |
| pathmnist | INT8 naive                  | msp ltt a=0.99           |        89.21 |         96.65 |      92.01 |      100    |    100    |
| pathmnist | INT8 naive                  | miss_risk plugin a=0.97  |        73.07 |         94.02 |      86.71 |       63.33 |     63.33 |
| pathmnist | INT8 naive                  | msp plugin a=0.97        |        75.79 |         93.98 |      88.97 |       56.67 |     56.67 |
| pathmnist | INT8 naive                  | miss_risk plugin a=0.99  |        81.15 |         95.82 |      89.44 |       50    |     50    |
| pathmnist | INT8 naive                  | msp plugin a=0.99        |        82.2  |         95.87 |      90.71 |       46.67 |     46.67 |
| pathmnist | INT8 naive, FP32 thresholds | always_small             |         0    |          0.09 |      43.73 |      nan    |    nan    |
| pathmnist | INT8 naive, FP32 thresholds | always_large             |       100    |         96.84 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk ltt a=0.97     |         4.36 |          6.68 |      46.75 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp ltt a=0.97           |        28.56 |         43.32 |      63.17 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk ltt a=0.99     |        55.56 |         71.1  |      77.92 |       13.33 |     13.33 |
| pathmnist | INT8 naive, FP32 thresholds | msp ltt a=0.99           |        71.02 |         85.31 |      86.15 |       23.33 |     23.33 |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk plugin a=0.97  |         0    |          0.09 |      43.73 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp plugin a=0.97        |         1    |          1.56 |      44.42 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | miss_risk plugin a=0.99  |         3.33 |          5.42 |      46.07 |        0    |      0    |
| pathmnist | INT8 naive, FP32 thresholds | msp plugin a=0.99        |        23.39 |         36.76 |      59.63 |        0    |      0    |
| pathmnist | INT8 mixed                  | always_small             |         0    |         94.12 |      91.52 |      nan    |    nan    |
| pathmnist | INT8 mixed                  | always_large             |       100    |         96.84 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 mixed                  | miss_risk ltt a=0.97     |         3.05 |         96.08 |      92.02 |      100    |     96.67 |
| pathmnist | INT8 mixed                  | msp ltt a=0.97           |         8.31 |         95.76 |      92.6  |      100    |     96.67 |
| pathmnist | INT8 mixed                  | miss_risk ltt a=0.99     |        21.35 |         96.97 |      92.98 |      100    |    100    |
| pathmnist | INT8 mixed                  | msp ltt a=0.99           |        28.19 |         96.79 |      93.11 |      100    |    100    |
| pathmnist | INT8 mixed                  | miss_risk plugin a=0.97  |         0.14 |         94.42 |      91.57 |       90    |     10    |
| pathmnist | INT8 mixed                  | msp plugin a=0.97        |         1.12 |         94.44 |      91.7  |       93.33 |     26.67 |
| pathmnist | INT8 mixed                  | miss_risk plugin a=0.99  |         2.43 |         95.89 |      91.97 |       60    |     16.67 |
| pathmnist | INT8 mixed                  | msp plugin a=0.99        |         8    |         95.7  |      92.47 |       50    |     23.33 |
| pathmnist | INT8 mixed, FP32 thresholds | always_small             |         0    |         94.12 |      91.52 |      nan    |    nan    |
| pathmnist | INT8 mixed, FP32 thresholds | always_large             |       100    |         96.84 |      92.94 |      nan    |    nan    |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.97     |         1.55 |         95.59 |      91.88 |      100    |     83.33 |
| pathmnist | INT8 mixed, FP32 thresholds | msp ltt a=0.97           |         5.76 |         95.33 |      92.4  |      100    |     86.67 |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.99     |        18.3  |         96.85 |      92.8  |      100    |     86.67 |
| pathmnist | INT8 mixed, FP32 thresholds | msp ltt a=0.99           |        25.48 |         96.62 |      93.01 |       93.33 |     86.67 |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.97  |         0.03 |         94.2  |      91.53 |       76.67 |      3.33 |
| pathmnist | INT8 mixed, FP32 thresholds | msp plugin a=0.97        |         0.13 |         94.17 |      91.54 |       76.67 |      3.33 |
| pathmnist | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.99  |         1.2  |         95.16 |      91.77 |       33.33 |      0    |
| pathmnist | INT8 mixed, FP32 thresholds | msp plugin a=0.99        |         5.07 |         95.12 |      92.3  |       26.67 |      3.33 |
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
| ham10000  | FP32                        | always_small             |         0    |         63.71 |      81.3  |      nan    |    nan    |
| ham10000  | FP32                        | always_large             |       100    |         64.79 |      82.81 |      nan    |    nan    |
| ham10000  | FP32                        | miss_risk ltt a=0.97     |       100    |         64.79 |      82.81 |      100    |    100    |
| ham10000  | FP32                        | msp ltt a=0.97           |       100    |         64.79 |      82.81 |      100    |    100    |
| ham10000  | FP32                        | miss_risk ltt a=0.99     |       100    |         64.79 |      82.81 |      100    |    100    |
| ham10000  | FP32                        | msp ltt a=0.99           |       100    |         64.79 |      82.81 |      100    |    100    |
| ham10000  | FP32                        | miss_risk plugin a=0.97  |         1.83 |         65.57 |      81.46 |       80    |      3.33 |
| ham10000  | FP32                        | msp plugin a=0.97        |         3.96 |         64.34 |      81.71 |       70    |      3.33 |
| ham10000  | FP32                        | miss_risk plugin a=0.99  |         2.07 |         65.81 |      81.49 |       60    |      0    |
| ham10000  | FP32                        | msp plugin a=0.99        |         6.35 |         64.46 |      81.85 |       50    |      3.33 |
| ham10000  | INT8 naive                  | always_small             |         0    |          0.1  |      64.66 |      nan    |    nan    |
| ham10000  | INT8 naive                  | always_large             |       100    |         65.28 |      82.74 |      nan    |    nan    |
| ham10000  | INT8 naive                  | miss_risk ltt a=0.97     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive                  | msp ltt a=0.97           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive                  | miss_risk ltt a=0.99     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive                  | msp ltt a=0.99           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive                  | miss_risk plugin a=0.97  |        95.51 |         62.27 |      82    |       46.67 |     46.67 |
| ham10000  | INT8 naive                  | msp plugin a=0.97        |        95.83 |         62.31 |      82.1  |       50    |     46.67 |
| ham10000  | INT8 naive                  | miss_risk plugin a=0.99  |        98.49 |         64.32 |      82.59 |       50    |     50    |
| ham10000  | INT8 naive                  | msp plugin a=0.99        |        98.36 |         64.17 |      82.53 |       60    |     60    |
| ham10000  | INT8 naive, FP32 thresholds | always_small             |         0    |          0.1  |      64.66 |      nan    |    nan    |
| ham10000  | INT8 naive, FP32 thresholds | always_large             |       100    |         65.28 |      82.74 |      nan    |    nan    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk ltt a=0.97     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | msp ltt a=0.97           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk ltt a=0.99     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | msp ltt a=0.99           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk plugin a=0.97  |         0.52 |          0.65 |      64.86 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | msp plugin a=0.97        |         1.86 |          1.3  |      65.29 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | miss_risk plugin a=0.99  |         0.63 |          0.65 |      64.88 |        0    |      0    |
| ham10000  | INT8 naive, FP32 thresholds | msp plugin a=0.99        |         4.41 |          3.03 |      65.8  |        0    |      0    |
| ham10000  | INT8 mixed                  | always_small             |         0    |         65.45 |      80.29 |      nan    |    nan    |
| ham10000  | INT8 mixed                  | always_large             |       100    |         65.28 |      82.74 |      nan    |    nan    |
| ham10000  | INT8 mixed                  | miss_risk ltt a=0.97     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed                  | msp ltt a=0.97           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed                  | miss_risk ltt a=0.99     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed                  | msp ltt a=0.99           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed                  | miss_risk plugin a=0.97  |         1.48 |         66.91 |      80.43 |       83.33 |      0    |
| ham10000  | INT8 mixed                  | msp plugin a=0.97        |         4.51 |         66.09 |      81.2  |       83.33 |      3.33 |
| ham10000  | INT8 mixed                  | miss_risk plugin a=0.99  |         2.42 |         67.98 |      80.61 |       80    |      0    |
| ham10000  | INT8 mixed                  | msp plugin a=0.99        |         7.11 |         66.36 |      81.55 |       76.67 |      0    |
| ham10000  | INT8 mixed, FP32 thresholds | always_small             |         0    |         65.45 |      80.29 |      nan    |    nan    |
| ham10000  | INT8 mixed, FP32 thresholds | always_large             |       100    |         65.28 |      82.74 |      nan    |    nan    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.97     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | msp ltt a=0.97           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk ltt a=0.99     |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | msp ltt a=0.99           |       100    |         65.28 |      82.74 |      100    |    100    |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.97  |         2.08 |         67.16 |      80.47 |       83.33 |      3.33 |
| ham10000  | INT8 mixed, FP32 thresholds | msp plugin a=0.97        |         4.85 |         65.78 |      80.98 |       83.33 |      6.67 |
| ham10000  | INT8 mixed, FP32 thresholds | miss_risk plugin a=0.99  |         2.42 |         67.69 |      80.56 |       80    |      3.33 |
| ham10000  | INT8 mixed, FP32 thresholds | msp plugin a=0.99        |         7.67 |         66.02 |      81.2  |       76.67 |      3.33 |
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