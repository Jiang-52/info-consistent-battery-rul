# Information-level v2 report

T = EOL cycle (SOH ≤ 0.7). L0 = raw t.

## Lifetimes

### nasa

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|
| B0005 | 168 | 161 | 167 | 0.222 |
| B0006 | 168 | 106 | 167 | 0.367 |
| B0007 | 168 | 167 | 167 | 0.182 |

### calce

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|
| CS2_35 | 882 | 349 | 881 | 1.665 |
| CS2_36 | 936 | 418 | 935 | 1.929 |
| CS2_37 | 972 | 410 | 971 | 2.011 |
| CS2_38 | 996 | 388 | 995 | 2.341 |

### severson

| Cell | n | EOL70 | T_record | regen_sum |
|---|---:|---:|---:|---:|

## Learned RUL RMSE (mean±std, seeds)

| Corpus | Model | Cell | L2 | L1 | L0 |
|---|---|---|---:|---:|---:|
| calce | mlp | CS2_35 | 2.73±1.38 | 3.00±2.15 | 33.42±2.78 |
| calce | mlp | CS2_36 | 2.20±1.30 | 2.28±1.12 | 40.33±2.14 |
| calce | mlp | CS2_37 | 2.72±1.00 | 2.86±1.11 | 45.28±2.45 |
| calce | mlp | CS2_38 | 11.87±1.95 | 12.40±1.60 | 46.00±3.99 |
| calce | patch-tf | CS2_35 | 24.85±12.20 | 32.00±4.12 | 23.80±3.12 |
| calce | patch-tf | CS2_36 | 33.74±9.75 | 41.34±0.95 | 27.05±6.08 |
| calce | patch-tf | CS2_37 | 34.65±7.25 | 35.74±0.78 | 26.11±10.75 |
| calce | patch-tf | CS2_38 | 41.56±12.26 | 25.31±9.90 | 28.93±2.13 |
| nasa | mlp | B0005 | 23.58±3.85 | 23.62±3.89 | 11.77±0.15 |
| nasa | mlp | B0006 | 10.62±1.69 | 10.52±1.70 | 6.46±0.99 |
| nasa | mlp | B0007 | 20.19±3.80 | 20.16±3.70 | 11.22±2.68 |
| nasa | patch-tf | B0005 | 19.17±2.43 | 19.17±2.42 | 10.21±1.21 |
| nasa | patch-tf | B0006 | 10.11±0.23 | 10.11±0.23 | 4.63±0.33 |
| nasa | patch-tf | B0007 | 15.25±2.27 | 15.26±2.27 | 10.84±2.08 |

## SOH RMSE

| Corpus | Model | Cell | L2 | L1 | L0 |
|---|---|---|---:|---:|---:|
| calce | mlp | CS2_35 | 0.1219±0.0652 | 0.1083±0.0216 | 0.1828±0.1032 |
| calce | mlp | CS2_36 | 0.2345±0.1210 | 0.2256±0.0355 | 0.2776±0.1371 |
| calce | mlp | CS2_37 | 0.1566±0.1152 | 0.1225±0.0288 | 0.3038±0.0936 |
| calce | mlp | CS2_38 | 0.1532±0.0591 | 0.2336±0.0210 | 0.3059±0.1880 |
| calce | patch-tf | CS2_35 | 1.2912±0.5235 | 2.7579±1.5481 | 1.0150±0.1851 |
| calce | patch-tf | CS2_36 | 3.6701±1.1277 | 6.1351±3.5455 | 2.3448±0.5467 |
| calce | patch-tf | CS2_37 | 4.0675±4.5061 | 4.3230±2.0411 | 1.5423±0.2325 |
| calce | patch-tf | CS2_38 | 1.4376±0.7108 | 3.6482±1.7495 | 1.2336±0.5560 |
| nasa | mlp | B0005 | 0.7063±0.1843 | 0.7007±0.1826 | 0.3302±0.0660 |
| nasa | mlp | B0006 | 0.8120±0.2939 | 0.8180±0.2988 | 0.2879±0.0462 |
| nasa | mlp | B0007 | 0.5625±0.1420 | 0.5640±0.1332 | 0.3358±0.0264 |
| nasa | patch-tf | B0005 | 0.3593±0.0567 | 0.3583±0.0565 | 0.5132±0.1256 |
| nasa | patch-tf | B0006 | 0.5564±0.3209 | 0.5548±0.3199 | 0.4427±0.0678 |
| nasa | patch-tf | B0007 | 0.4028±0.0108 | 0.4044±0.0119 | 0.6538±0.4076 |

## Classical RUL RMSE

| Method | Corpus | Cell | RMSE |
|---|---|---|---:|
| persist_fade | nasa | B0005 | 7.63 |
| linear | nasa | B0005 | 7.92 |
| persist_recursive | nasa | B0005 | 6.14 |
| persist_hold | nasa | B0005 | 1.00 |
| persist_mean | nasa | B0005 | 50.61 |
| persist_fade | nasa | B0006 | 3.68 |
| linear | nasa | B0006 | 7.33 |
| persist_recursive | nasa | B0006 | 0.48 |
| persist_hold | nasa | B0006 | 1.00 |
| persist_mean | nasa | B0006 | 25.81 |
| persist_fade | nasa | B0007 | 30.29 |
| linear | nasa | B0007 | 28.10 |
| persist_recursive | nasa | B0007 | 15.53 |
| persist_hold | nasa | B0007 | 1.00 |
| persist_mean | nasa | B0007 | 51.61 |
| persist_fade | calce | CS2_35 | 24684.31 |
| linear | calce | CS2_35 | 123.37 |
| persist_recursive | calce | CS2_35 | 5.07 |
| persist_hold | calce | CS2_35 | 1.00 |
| persist_mean | calce | CS2_35 | 135.53 |
| persist_fade | calce | CS2_36 | 20511.36 |
| linear | calce | CS2_36 | 47.55 |
| persist_recursive | calce | CS2_36 | 262.71 |
| persist_hold | calce | CS2_36 | 1.00 |
| persist_mean | calce | CS2_36 | 166.59 |
| persist_fade | calce | CS2_37 | 47121.18 |
| linear | calce | CS2_37 | 79.85 |
| persist_recursive | calce | CS2_37 | 7.83 |
| persist_hold | calce | CS2_37 | 1.00 |
| persist_mean | calce | CS2_37 | 162.84 |
| persist_fade | calce | CS2_38 | 48778.34 |
| linear | calce | CS2_38 | 308.31 |
| persist_recursive | calce | CS2_38 | 265.78 |
| persist_hold | calce | CS2_38 | 1.00 |
| persist_mean | calce | CS2_38 | 153.06 |

## Seed-level tests (not cell-level n)

| Contrast | Corpus | Model | Cell | ΔRMSE | Cohen d | Wilcoxon p | n |
|---|---|---|---|---:|---:|---:|---:|
| L2-vs-L0 | calce | mlp | CS2_35 | 30.69 | 13.98 | 0.2500 | 3 |
| L2-vs-L1 | calce | mlp | CS2_35 | 0.27 | 0.15 | 0.7500 | 3 |
| L1-vs-L0 | calce | mlp | CS2_35 | 30.41 | 12.24 | 0.2500 | 3 |
| L2-vs-L0 | calce | mlp | CS2_36 | 38.13 | 21.52 | 0.2500 | 3 |
| L2-vs-L1 | calce | mlp | CS2_36 | 0.08 | 0.07 | 0.7500 | 3 |
| L1-vs-L0 | calce | mlp | CS2_36 | 38.05 | 22.26 | 0.2500 | 3 |
| L2-vs-L0 | calce | mlp | CS2_37 | 42.55 | 22.73 | 0.2500 | 3 |
| L2-vs-L1 | calce | mlp | CS2_37 | 0.14 | 0.13 | 0.5000 | 3 |
| L1-vs-L0 | calce | mlp | CS2_37 | 42.41 | 22.31 | 0.2500 | 3 |
| L2-vs-L0 | calce | mlp | CS2_38 | 34.13 | 10.86 | 0.2500 | 3 |
| L2-vs-L1 | calce | mlp | CS2_38 | 0.53 | 0.30 | 0.5000 | 3 |
| L1-vs-L0 | calce | mlp | CS2_38 | 33.60 | 11.04 | 0.2500 | 3 |
| L2-vs-L0 | calce | patch-tf | CS2_35 | -1.05 | -0.12 | 1.0000 | 3 |
| L2-vs-L1 | calce | patch-tf | CS2_35 | 7.15 | 0.79 | 0.7500 | 3 |
| L1-vs-L0 | calce | patch-tf | CS2_35 | -8.20 | -2.24 | 0.2500 | 3 |
| L2-vs-L0 | calce | patch-tf | CS2_36 | -6.69 | -0.82 | 0.5000 | 3 |
| L2-vs-L1 | calce | patch-tf | CS2_36 | 7.60 | 1.10 | 0.2500 | 3 |
| L1-vs-L0 | calce | patch-tf | CS2_36 | -14.29 | -3.28 | 0.2500 | 3 |
| L2-vs-L0 | calce | patch-tf | CS2_37 | -8.54 | -0.93 | 0.7500 | 3 |
| L2-vs-L1 | calce | patch-tf | CS2_37 | 1.09 | 0.21 | 1.0000 | 3 |
| L1-vs-L0 | calce | patch-tf | CS2_37 | -9.63 | -1.26 | 0.2500 | 3 |
| L2-vs-L0 | calce | patch-tf | CS2_38 | -12.62 | -1.44 | 0.2500 | 3 |
| L2-vs-L1 | calce | patch-tf | CS2_38 | -16.25 | -1.46 | 0.2500 | 3 |
| L1-vs-L0 | calce | patch-tf | CS2_38 | 3.63 | 0.51 | 0.7500 | 3 |
| L2-vs-L0 | nasa | mlp | B0005 | -11.81 | -4.33 | 0.2500 | 3 |
| L2-vs-L1 | nasa | mlp | B0005 | 0.04 | 0.01 | 0.7500 | 3 |
| L1-vs-L0 | nasa | mlp | B0005 | -11.85 | -4.31 | 0.2500 | 3 |
| L2-vs-L0 | nasa | mlp | B0006 | -4.16 | -3.00 | 0.2500 | 3 |
| L2-vs-L1 | nasa | mlp | B0006 | -0.10 | -0.06 | 0.2500 | 3 |
| L1-vs-L0 | nasa | mlp | B0006 | -4.06 | -2.91 | 0.2500 | 3 |
| L2-vs-L0 | nasa | mlp | B0007 | -8.97 | -2.73 | 0.2500 | 3 |
| L2-vs-L1 | nasa | mlp | B0007 | -0.03 | -0.01 | 1.0000 | 3 |
| L1-vs-L0 | nasa | mlp | B0007 | -8.94 | -2.77 | 0.2500 | 3 |
| L2-vs-L0 | nasa | patch-tf | B0005 | -8.95 | -4.67 | 0.2500 | 3 |
| L2-vs-L1 | nasa | patch-tf | B0005 | 0.01 | 0.00 | 1.0000 | 3 |
| L1-vs-L0 | nasa | patch-tf | B0005 | -8.96 | -4.69 | 0.2500 | 3 |
| L2-vs-L0 | nasa | patch-tf | B0006 | -5.49 | -19.33 | 0.2500 | 3 |
| L2-vs-L1 | nasa | patch-tf | B0006 | -0.01 | -0.03 | 0.7500 | 3 |
| L1-vs-L0 | nasa | patch-tf | B0006 | -5.48 | -19.42 | 0.2500 | 3 |
| L2-vs-L0 | nasa | patch-tf | B0007 | -4.41 | -2.03 | 0.2500 | 3 |
| L2-vs-L1 | nasa | patch-tf | B0007 | 0.02 | 0.01 | 0.5000 | 3 |
| L1-vs-L0 | nasa | patch-tf | B0007 | -4.43 | -2.03 | 0.2500 | 3 |
