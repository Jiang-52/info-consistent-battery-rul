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
| severson | mlp | b1c15 | 4.91±1.93 | 27.69±3.65 | 215.05±10.40 |
| severson | mlp | b1c30 | 13.03±2.17 | 44.03±2.02 | 232.00±15.42 |
| severson | mlp | b1c32 | 8.42±1.10 | 13.33±1.34 | 225.25±21.10 |
| severson | mlp | b1c34 | 12.96±5.92 | 47.17±2.17 | 213.51±21.21 |
| severson | mlp | b1c35 | 5.93±2.29 | 16.22±9.20 | 242.03±23.56 |
| severson | mlp | b1c36 | 6.70±0.17 | 14.37±3.39 | 216.49±10.22 |
| severson | mlp | b2c0 | 13.70±1.49 | 14.28±2.65 | 408.09±12.25 |
| severson | mlp | b2c1 | 158.22±89.76 | 17.08±9.36 | 557.65±10.94 |
| severson | mlp | b2c2 | 8.06±1.14 | 20.43±5.05 | 334.12±14.52 |
| severson | mlp | b2c3 | 10.39±2.77 | 10.27±3.81 | 388.01±7.08 |
| severson | mlp | b2c4 | 7.85±1.03 | 8.43±1.33 | 329.56±15.10 |
| severson | mlp | b2c41 | 12.11±3.73 | 23.92±4.13 | 349.61±9.91 |
| severson | mlp | b2c44 | 4.76±3.32 | 30.49±2.80 | 328.37±9.33 |
| severson | mlp | b2c46 | 6.46±3.62 | 48.69±4.64 | 352.70±21.09 |
| severson | mlp | b2c47 | 7.96±2.36 | 13.46±1.53 | 218.97±23.00 |
| severson | mlp | b3c16 | 16.60±8.16 | 36.58±16.34 | 151.21±16.95 |
| severson | mlp | b3c17 | 11.79±4.23 | 86.67±34.34 | 131.00±8.24 |
| severson | mlp | b3c31 | 12.47±2.77 | 56.44±13.56 | 230.25±6.55 |
| severson | mlp | b3c33 | 17.38±3.66 | 143.24±28.46 | 130.84±14.16 |
| severson | mlp | b3c34 | 11.67±5.89 | 48.33±12.68 | 143.12±4.47 |
| severson | mlp | b3c38 | 18.99±1.02 | 39.27±20.30 | 121.22±5.51 |
| severson | mlp | b3c39 | 13.62±4.33 | 75.32±47.20 | 155.05±8.01 |
| severson | mlp | b3c45 | 15.42±3.59 | 38.97±26.93 | 133.84±8.67 |
| severson | mlp | b3c7 | 21.57±4.74 | 37.22±16.97 | 124.92±20.93 |

## SOH RMSE

| Corpus | Model | Cell | L2 | L1 | L0 |
|---|---|---|---:|---:|---:|
| severson | mlp | b1c15 | 0.0082±0.0054 | 0.0101±0.0032 | 0.0615±0.0050 |
| severson | mlp | b1c30 | 0.0135±0.0116 | 0.0118±0.0037 | 0.0685±0.0074 |
| severson | mlp | b1c32 | 0.0108±0.0021 | 0.0089±0.0022 | 0.0737±0.0069 |
| severson | mlp | b1c34 | 0.0119±0.0030 | 0.0054±0.0007 | 0.0634±0.0107 |
| severson | mlp | b1c35 | 0.0119±0.0025 | 0.0135±0.0067 | 0.0692±0.0125 |
| severson | mlp | b1c36 | 0.0195±0.0159 | 0.0150±0.0098 | 0.0640±0.0037 |
| severson | mlp | b2c0 | 0.0522±0.0335 | 0.0121±0.0064 | 0.1198±0.0049 |
| severson | mlp | b2c1 | 0.1237±0.0149 | 0.0395±0.0207 | 0.1459±0.0139 |
| severson | mlp | b2c2 | 0.0310±0.0095 | 0.0068±0.0026 | 0.0794±0.0046 |
| severson | mlp | b2c3 | 0.0570±0.0219 | 0.0122±0.0075 | 0.1095±0.0020 |
| severson | mlp | b2c4 | 0.0218±0.0193 | 0.0115±0.0049 | 0.0956±0.0009 |
| severson | mlp | b2c41 | 0.0220±0.0069 | 0.0169±0.0057 | 0.0778±0.0045 |
| severson | mlp | b2c44 | 0.0179±0.0009 | 0.0105±0.0027 | 0.0847±0.0014 |
| severson | mlp | b2c46 | 0.0230±0.0037 | 0.0180±0.0072 | 0.0765±0.0032 |
| severson | mlp | b2c47 | 0.0179±0.0055 | 0.0155±0.0046 | 0.0879±0.0087 |
| severson | mlp | b3c16 | 0.0119±0.0075 | 0.0164±0.0085 | 0.0698±0.0123 |
| severson | mlp | b3c17 | 0.0131±0.0022 | 0.0210±0.0064 | 0.0767±0.0140 |
| severson | mlp | b3c31 | 0.0075±0.0033 | 0.0070±0.0009 | 0.0596±0.0088 |
| severson | mlp | b3c33 | 0.0171±0.0130 | 0.0142±0.0011 | 0.0514±0.0026 |
| severson | mlp | b3c34 | 0.0110±0.0025 | 0.0156±0.0046 | 0.0665±0.0132 |
| severson | mlp | b3c38 | 0.0113±0.0033 | 0.0172±0.0086 | 0.0639±0.0120 |
| severson | mlp | b3c39 | 0.0115±0.0020 | 0.0180±0.0014 | 0.0531±0.0107 |
| severson | mlp | b3c45 | 0.0105±0.0090 | 0.0138±0.0073 | 0.0673±0.0130 |
| severson | mlp | b3c7 | 0.0086±0.0036 | 0.0264±0.0113 | 0.0553±0.0014 |

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
| L2-vs-L0 | severson | mlp | b1c15 | 210.14 | 28.09 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c15 | 22.78 | 7.80 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c15 | 187.36 | 24.04 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c30 | 218.98 | 19.89 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c30 | 31.01 | 14.78 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c30 | 187.97 | 17.10 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c32 | 216.83 | 14.51 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c32 | 4.91 | 4.00 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c32 | 211.91 | 14.17 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c34 | 200.55 | 12.88 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c34 | 34.20 | 7.67 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c34 | 166.34 | 11.04 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c35 | 236.10 | 14.11 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c35 | 10.29 | 1.53 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c35 | 225.81 | 12.63 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b1c36 | 209.79 | 29.02 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b1c36 | 7.68 | 3.20 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b1c36 | 202.11 | 26.54 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c0 | 394.39 | 45.18 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c0 | 0.58 | 0.27 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b2c0 | 393.82 | 44.42 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c1 | 399.43 | 6.25 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c1 | -141.14 | -2.21 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c1 | 540.58 | 53.12 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c2 | 326.06 | 31.66 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c2 | 12.36 | 3.38 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c2 | 313.69 | 28.86 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c3 | 377.61 | 70.24 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c3 | -0.12 | -0.04 | 1.0000 | 3 |
| L1-vs-L0 | severson | mlp | b2c3 | 377.74 | 66.42 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c4 | 321.70 | 30.07 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c4 | 0.58 | 0.49 | 0.7500 | 3 |
| L1-vs-L0 | severson | mlp | b2c4 | 321.12 | 29.97 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c41 | 337.50 | 45.09 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c41 | 11.80 | 3.00 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c41 | 325.70 | 42.92 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c44 | 323.62 | 46.23 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c44 | 25.73 | 8.38 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c44 | 297.89 | 43.26 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c46 | 346.24 | 22.88 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c46 | 42.23 | 10.15 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c46 | 304.00 | 19.91 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b2c47 | 211.01 | 12.90 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b2c47 | 5.50 | 2.76 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b2c47 | 205.51 | 12.61 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c16 | 134.61 | 10.12 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c16 | 19.98 | 1.55 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b3c16 | 114.63 | 6.89 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c17 | 119.21 | 18.20 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c17 | 74.88 | 3.06 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c17 | 44.33 | 1.78 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c31 | 217.78 | 43.31 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c31 | 43.97 | 4.49 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c31 | 173.81 | 16.32 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c33 | 113.46 | 10.97 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c33 | 125.85 | 6.20 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c33 | -12.40 | -0.55 | 0.7500 | 3 |
| L2-vs-L0 | severson | mlp | b3c34 | 131.45 | 25.12 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c34 | 36.65 | 3.71 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c34 | 94.79 | 9.97 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c38 | 102.23 | 25.80 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c38 | 20.28 | 1.41 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b3c38 | 81.95 | 5.51 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c39 | 141.43 | 21.96 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c39 | 61.69 | 1.84 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c39 | 79.74 | 2.36 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c45 | 118.41 | 17.84 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c45 | 23.55 | 1.23 | 0.5000 | 3 |
| L1-vs-L0 | severson | mlp | b3c45 | 94.86 | 4.74 | 0.2500 | 3 |
| L2-vs-L0 | severson | mlp | b3c7 | 103.35 | 6.81 | 0.2500 | 3 |
| L2-vs-L1 | severson | mlp | b3c7 | 15.65 | 1.26 | 0.2500 | 3 |
| L1-vs-L0 | severson | mlp | b3c7 | 87.70 | 4.60 | 0.2500 | 3 |
