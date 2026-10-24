# Information-level v2 report

T = EOL cycle (SOH ≤ 0.7). L0 = raw t.

## Lifetimes

### nasa

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|

### calce

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|

### severson

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|
| b2c1 | 170 | 147 | 169 | 0.007 |
| b2c0 | 326 | 299 | 325 | 0.077 |
| b2c3 | 361 | 334 | 360 | 0.042 |
| b2c41 | 452 | 428 | 451 | 0.038 |
| b2c46 | 449 | 428 | 448 | 0.019 |
| b2c2 | 464 | 437 | 463 | 0.051 |
| b2c4 | 471 | 443 | 470 | 0.116 |
| b2c44 | 479 | 456 | 478 | 0.255 |
| b1c35 | 702 | 701 | 701 | 0.038 |
| b1c36 | 703 | 702 | 702 | 0.043 |
| b1c30 | 708 | 707 | 707 | 0.048 |
| b2c47 | 745 | 712 | 744 | 0.027 |
| b1c15 | 718 | 717 | 717 | 0.034 |
| b1c32 | 730 | 729 | 729 | 0.028 |
| b3c31 | 730 | 729 | 729 | 0.014 |
| b1c34 | 741 | 740 | 740 | 0.050 |
| b3c39 | 1155 | 1154 | 1154 | 0.010 |
| b3c34 | 1157 | 1156 | 1156 | 0.011 |
| b3c33 | 1283 | 1282 | 1282 | 0.009 |
| b3c17 | 1314 | 1313 | 1313 | 0.026 |
| b3c16 | 1637 | 1636 | 1636 | 0.053 |
| b3c45 | 1800 | 1799 | 1799 | 0.082 |
| b3c7 | 1835 | 1834 | 1834 | 0.104 |
| b3c38 | 1934 | 1933 | 1933 | 0.013 |

## Learned RUL RMSE (mean±std, seeds)

| Corpus | Model | Cell | L2 | L1 | L0 |
|---|---|---|---:|---:|---:|
| severson | mlp | b1c15 | 19.28±0.48 | 19.31±0.43 | 74.85±12.06 |
| severson | mlp | b1c30 | 16.18±1.80 | 15.25±0.64 | 71.88±8.07 |
| severson | mlp | b1c32 | 19.57±0.37 | 19.70±0.66 | 65.59±14.23 |
| severson | mlp | b1c34 | 18.85±1.14 | 18.59±1.84 | 69.82±18.77 |
| severson | mlp | b1c35 | 15.55±4.58 | 18.13±2.29 | 75.24±17.07 |
| severson | mlp | b1c36 | 19.24±1.83 | 17.50±0.38 | 61.81±4.44 |
| severson | mlp | b2c0 | 1.37±0.71 | 0.99±0.60 | 32.09±5.57 |
| severson | mlp | b2c1 | 2.18±0.69 | 8.75±7.18 | 13.24±1.24 |
| severson | mlp | b2c2 | 5.15±1.40 | 7.09±0.30 | 45.98±4.70 |
| severson | mlp | b2c3 | 1.44±0.88 | 1.45±0.75 | 34.73±7.14 |
| severson | mlp | b2c4 | 4.40±1.43 | 4.92±1.36 | 49.01±7.90 |
| severson | mlp | b2c41 | 5.52±0.77 | 6.34±0.74 | 43.60±3.52 |
| severson | mlp | b2c44 | 6.68±0.35 | 9.44±1.02 | 53.40±10.98 |
| severson | mlp | b2c46 | 5.72±0.70 | 6.38±1.09 | 46.18±12.11 |
| severson | mlp | b2c47 | 7.20±1.76 | 7.78±0.95 | 79.61±2.64 |
| severson | mlp | b3c16 | 36.55±2.84 | 38.70±3.26 | 132.60±21.78 |
| severson | mlp | b3c17 | 34.12±6.70 | 36.52±7.21 | 138.86±13.88 |
| severson | mlp | b3c31 | 11.58±1.86 | 11.49±2.17 | 76.48±4.01 |
| severson | mlp | b3c33 | 22.98±5.37 | 21.00±7.50 | 110.90±11.36 |
| severson | mlp | b3c34 | 31.08±3.47 | 27.92±6.20 | 116.89±24.92 |
| severson | mlp | b3c38 | 45.86±4.19 | 51.48±14.16 | 159.64±19.26 |
| severson | mlp | b3c39 | 13.24±1.42 | 16.34±6.42 | 104.48±2.04 |
| severson | mlp | b3c45 | 75.63±45.35 | 40.22±7.89 | 174.03±27.87 |
| severson | mlp | b3c7 | 44.55±5.51 | 40.54±11.08 | 165.16±9.61 |

## SOH RMSE

| Corpus | Model | Cell | L2 | L1 | L0 |
|---|---|---|---:|---:|---:|
| severson | mlp | b1c15 | 0.9260±0.7705 | 0.9163±0.6475 | 0.3867±0.1293 |
| severson | mlp | b1c30 | 0.9617±0.2072 | 0.8663±0.3196 | 0.3101±0.0875 |
| severson | mlp | b1c32 | 0.8917±0.1462 | 0.8952±0.2204 | 0.2212±0.0915 |
| severson | mlp | b1c34 | 0.6127±0.2361 | 0.7862±0.4270 | 0.3722±0.0593 |
| severson | mlp | b1c35 | 0.7687±0.1435 | 0.8872±0.3914 | 0.2003±0.1200 |
| severson | mlp | b1c36 | 0.7780±0.1648 | 0.7545±0.1071 | 0.3421±0.2264 |
| severson | mlp | b2c0 | 0.3194±0.2435 | 0.2233±0.0902 | 0.2049±0.0439 |
| severson | mlp | b2c1 | 0.1145±0.0227 | 0.0907±0.0466 | 0.2651±0.1382 |
| severson | mlp | b2c2 | 0.6567±0.0973 | 1.0317±0.4203 | 0.2544±0.1061 |
| severson | mlp | b2c3 | 0.3843±0.3981 | 0.3846±0.4191 | 0.2557±0.1886 |
| severson | mlp | b2c4 | 0.4756±0.2353 | 0.4491±0.1703 | 0.2247±0.1027 |
| severson | mlp | b2c41 | 0.6539±0.4721 | 0.6713±0.3575 | 0.2269±0.0715 |
| severson | mlp | b2c44 | 0.5400±0.3801 | 0.6235±0.2995 | 0.2825±0.0336 |
| severson | mlp | b2c46 | 0.6759±0.3760 | 0.5667±0.2909 | 0.2531±0.1154 |
| severson | mlp | b2c47 | 0.4389±0.3042 | 0.5441±0.3255 | 0.2736±0.2282 |
| severson | mlp | b3c16 | 1.3460±0.3864 | 0.4442±0.3248 | 0.2280±0.1331 |
| severson | mlp | b3c17 | 1.2713±0.7843 | 1.1376±0.9567 | 0.1905±0.1073 |
| severson | mlp | b3c31 | 0.7515±0.3833 | 0.7711±0.3312 | 0.2984±0.0715 |
| severson | mlp | b3c33 | 0.8593±0.2600 | 0.8857±0.4260 | 0.2328±0.1125 |
| severson | mlp | b3c34 | 1.5589±0.5878 | 1.1483±1.0915 | 0.2129±0.1462 |
| severson | mlp | b3c38 | 1.5035±0.4547 | 0.8283±0.5128 | 0.2677±0.1409 |
| severson | mlp | b3c39 | 1.0214±0.3003 | 0.6081±0.0994 | 0.1975±0.1079 |
| severson | mlp | b3c45 | 2.5240±1.6396 | 0.5985±0.4145 | 0.2270±0.1478 |
| severson | mlp | b3c7 | 0.8650±0.7396 | 0.5983±0.4219 | 0.0961±0.0428 |

## Classical RUL RMSE

| Method | Corpus | Cell | RMSE |
|---|---|---|---:|
| persist_fade | severson | b2c1 | 65.28 |
| linear | severson | b2c1 | 67.23 |
| persist_recursive | severson | b2c1 | 75.90 |
| persist_hold | severson | b2c1 | 1.00 |
| persist_mean | severson | b2c1 | 44.34 |
| persist_fade | severson | b2c0 | 135.50 |
| linear | severson | b2c0 | 137.79 |
| persist_recursive | severson | b2c0 | 139.99 |
| persist_hold | severson | b2c0 | 1.00 |
| persist_mean | severson | b2c0 | 112.98 |
| persist_fade | severson | b2c3 | 970.29 |
| linear | severson | b2c3 | 170.43 |
| persist_recursive | severson | b2c3 | 206.19 |
| persist_hold | severson | b2c3 | 1.00 |
| persist_mean | severson | b2c3 | 128.76 |
| persist_fade | severson | b2c41 | 156.03 |
| linear | severson | b2c41 | 160.32 |
| persist_recursive | severson | b2c41 | 304.50 |
| persist_hold | severson | b2c41 | 1.00 |
| persist_mean | severson | b2c41 | 171.10 |
| persist_fade | severson | b2c46 | 179.63 |
| linear | severson | b2c46 | 186.82 |
| persist_recursive | severson | b2c46 | 366.23 |
| persist_hold | severson | b2c46 | 1.00 |
| persist_mean | severson | b2c46 | 171.10 |
| persist_fade | severson | b2c2 | 171.67 |
| linear | severson | b2c2 | 176.41 |
| persist_recursive | severson | b2c2 | 318.07 |
| persist_hold | severson | b2c2 | 1.00 |
| persist_mean | severson | b2c2 | 175.11 |
| persist_fade | severson | b2c4 | 166.23 |
| linear | severson | b2c4 | 169.39 |
| persist_recursive | severson | b2c4 | 263.71 |
| persist_hold | severson | b2c4 | 1.00 |
| persist_mean | severson | b2c4 | 177.87 |
| persist_fade | severson | b2c44 | 163.99 |
| linear | severson | b2c44 | 170.46 |
| persist_recursive | severson | b2c44 | 345.78 |
| persist_hold | severson | b2c44 | 1.00 |
| persist_mean | severson | b2c44 | 183.63 |
| persist_fade | severson | b1c35 | 190.98 |
| linear | severson | b1c35 | 192.96 |
| persist_recursive | severson | b1c35 | 401.22 |
| persist_hold | severson | b1c35 | 1.00 |
| persist_mean | severson | b1c35 | 292.35 |
| persist_fade | severson | b1c36 | 178.59 |
| linear | severson | b1c36 | 180.45 |
| persist_recursive | severson | b1c36 | 355.76 |
| persist_hold | severson | b1c36 | 1.00 |
| persist_mean | severson | b1c36 | 292.85 |
| persist_fade | severson | b1c30 | 223.98 |
| linear | severson | b1c30 | 227.35 |
| persist_recursive | severson | b1c30 | 493.28 |
| persist_hold | severson | b1c30 | 1.00 |
| persist_mean | severson | b1c30 | 295.11 |
| persist_fade | severson | b2c47 | 310.17 |
| linear | severson | b2c47 | 313.66 |
| persist_recursive | severson | b2c47 | 556.17 |
| persist_hold | severson | b2c47 | 1.00 |
| persist_mean | severson | b2c47 | 299.12 |
| persist_fade | severson | b1c15 | 224.94 |
| linear | severson | b1c15 | 229.09 |
| persist_recursive | severson | b1c15 | 505.44 |
| persist_hold | severson | b1c15 | 1.00 |
| persist_mean | severson | b1c15 | 299.62 |
| persist_fade | severson | b1c32 | 188.62 |
| linear | severson | b1c32 | 191.78 |
| persist_recursive | severson | b1c32 | 405.59 |
| persist_hold | severson | b1c32 | 1.00 |
| persist_mean | severson | b1c32 | 304.88 |
| persist_fade | severson | b3c31 | 377.84 |
| linear | severson | b3c31 | 383.70 |
| persist_recursive | severson | b3c31 | 766.96 |
| persist_hold | severson | b3c31 | 1.00 |
| persist_mean | severson | b3c31 | 304.88 |
| persist_fade | severson | b1c34 | 260.62 |
| linear | severson | b1c34 | 261.79 |
| persist_recursive | severson | b1c34 | 554.20 |
| persist_hold | severson | b1c34 | 1.00 |
| persist_mean | severson | b1c34 | 309.89 |
| persist_fade | severson | b3c39 | 870.79 |
| linear | severson | b3c39 | 875.77 |
| persist_recursive | severson | b3c39 | 1750.02 |
| persist_hold | severson | b3c39 | 1.00 |
| persist_mean | severson | b3c39 | 496.53 |
| persist_fade | severson | b3c34 | 402.36 |
| linear | severson | b3c34 | 408.24 |
| persist_recursive | severson | b3c34 | 1133.82 |
| persist_hold | severson | b3c34 | 1.00 |
| persist_mean | severson | b3c34 | 497.52 |
| persist_fade | severson | b3c33 | 784.77 |
| linear | severson | b3c33 | 789.34 |
| persist_recursive | severson | b3c33 | 1745.73 |
| persist_hold | severson | b3c33 | 1.00 |
| persist_mean | severson | b3c33 | 554.39 |
| persist_fade | severson | b3c17 | 355.71 |
| linear | severson | b3c17 | 356.87 |
| persist_recursive | severson | b3c17 | 795.42 |
| persist_hold | severson | b3c17 | 1.00 |
| persist_mean | severson | b3c17 | 568.41 |
| persist_fade | severson | b3c16 | 525.08 |
| linear | severson | b3c16 | 531.54 |
| persist_recursive | severson | b3c16 | 1433.25 |
| persist_hold | severson | b3c16 | 1.00 |
| persist_mean | severson | b3c16 | 713.97 |
| persist_fade | severson | b3c45 | 563.96 |
| linear | severson | b3c45 | 567.33 |
| persist_recursive | severson | b3c45 | 1388.74 |
| persist_hold | severson | b3c45 | 1.00 |
| persist_mean | severson | b3c45 | 787.37 |
| persist_fade | severson | b3c7 | 523.17 |
| linear | severson | b3c7 | 523.40 |
| persist_recursive | severson | b3c7 | 986.11 |
| persist_hold | severson | b3c7 | 1.00 |
| persist_mean | severson | b3c7 | 803.15 |
| persist_fade | severson | b3c38 | 567.75 |
| linear | severson | b3c38 | 568.02 |
| persist_recursive | severson | b3c38 | 1012.16 |
| persist_hold | severson | b3c38 | 1.00 |
| persist_mean | severson | b3c38 | 847.99 |

## Seed-level tests (not cell-level n)

| Contrast | Corpus | Model | Cell | ΔRMSE | Cohen d | Wilcoxon p | n |
|---|---|---|---|---:|---:|---:|---:|
| L2-vs-L0 | severson | mlp | b1c15 | 55.57 | 6.51 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c15 | 0.03 | 0.06 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b1c15 | 55.54 | 6.51 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c30 | 55.70 | 9.53 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c30 | -0.93 | -0.69 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b1c30 | 56.63 | 9.90 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c32 | 46.02 | 4.57 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c32 | 0.13 | 0.23 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b1c32 | 45.89 | 4.56 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c34 | 50.97 | 3.83 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c34 | -0.26 | -0.17 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b1c34 | 51.23 | 3.84 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c35 | 59.68 | 4.78 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c35 | 2.58 | 0.71 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b1c35 | 57.10 | 4.69 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c36 | 42.57 | 12.54 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c36 | -1.74 | -1.32 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c36 | 44.32 | 14.06 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c0 | 30.72 | 7.73 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c0 | -0.38 | -0.58 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b2c0 | 31.10 | 7.84 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c1 | 11.06 | 11.07 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c1 | 6.57 | 1.29 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c1 | 4.49 | 0.87 | 0.5000 | 3 |
| L2-vs-L0 | severson | mlp | b2c2 | 40.83 | 11.77 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c2 | 1.94 | 1.92 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c2 | 38.89 | 11.67 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c3 | 33.29 | 6.55 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c3 | 0.01 | 0.02 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b2c3 | 33.28 | 6.56 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c4 | 44.61 | 7.86 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c4 | 0.52 | 0.37 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b2c4 | 44.10 | 7.78 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c41 | 38.08 | 14.95 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c41 | 0.82 | 1.08 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c41 | 37.27 | 14.66 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c44 | 46.72 | 6.01 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c44 | 2.76 | 3.62 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c44 | 43.96 | 5.64 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c46 | 40.45 | 4.72 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c46 | 0.66 | 0.72 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c46 | 39.80 | 4.63 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c47 | 72.41 | 32.30 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c47 | 0.58 | 0.41 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b2c47 | 71.83 | 36.21 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c16 | 96.05 | 6.18 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c16 | 2.15 | 0.70 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c16 | 93.89 | 6.03 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c17 | 104.74 | 9.61 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c17 | 2.40 | 0.35 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c17 | 102.34 | 9.25 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c31 | 64.90 | 20.77 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c31 | -0.09 | -0.04 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c31 | 64.99 | 20.15 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c33 | 87.92 | 9.90 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c33 | -1.98 | -0.30 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b3c33 | 89.90 | 9.34 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c34 | 85.82 | 4.82 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c34 | -3.15 | -0.63 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c34 | 88.97 | 4.90 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c38 | 113.78 | 8.16 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c38 | 5.62 | 0.54 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c38 | 108.16 | 6.40 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c39 | 91.24 | 51.84 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c39 | 3.10 | 0.67 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b3c39 | 88.14 | 18.50 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c45 | 98.40 | 2.61 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c45 | -35.42 | -1.09 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b3c45 | 133.82 | 6.53 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c7 | 120.61 | 15.40 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c7 | -4.02 | -0.46 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b3c7 | 124.62 | 12.02 | 0.2500 | 3 |
