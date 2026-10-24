# TCN Battery SOH Estimation Baseline

基于 TCN 的 NASA 锂电池 SOH 估计基线（只做 SOH），参考开源实现缝合：

- [Bansilal01 Cycle-Based SoH](https://github.com/Bansilal01/Cycle-Based-State-of-Health-SoH-Estimation-Using-MLP-GRU-TCN-and-LSTM) — `.mat` 解析 / SimpleTCN
- [EV-Degradation-Engine](https://github.com/HassanSidd0946/EV-Degradation-Engine) — 50-cycle 滑动窗口思路

## 快速开始

```powershell
# 1. 安装依赖
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
py -m pip install -r requirements.txt

# 2. 下载 NASA 数据 (B0005/B0006/B0007)
py scripts/download_nasa.py

# 3. 训练 TCN SOH（推荐：50-cycle 历史窗口 -> 下一时刻 SOH）
py scripts/train_soh_tcn.py --all --mode window --window-size 50 --epochs 400

# 4. 对照：SimpleTCN 曲线拟合
py scripts/train_soh_tcn.py --battery B0005 --mode simple --epochs 3000
```

## Baseline 结果（window=50, epochs=400, CPU）

| Battery | Test RMSE | Test MAE | Full RMSE | Full MAE | Full R² |
|---------|-----------|----------|-----------|----------|---------|
| B0005   | 0.0107    | 0.0095   | 0.0120    | 0.0098   | 0.9717  |
| B0006   | 0.0248    | 0.0210   | 0.0208    | 0.0168   | 0.9092  |
| B0007   | 0.0073    | 0.0061   | 0.0106    | 0.0080   | 0.9614  |

说明：按时间顺序 8:2 划分时，测试段多为寿命末期、SOH 波动很小，Test R² 容易偏低；**Full** 指标为全寿命一阶预测曲线，更适合看整体拟合效果。曲线图见 `outputs/soh_b000x.png`。

## 输出

- `outputs/soh_b0005.png` — SOH 预测曲线
- `outputs/metrics_b0005.json` — RMSE / MAE / R²
- `outputs/tcn_b0005.pt` — 模型权重
- `outputs/metrics_summary.json` — 汇总

## 数据说明

NASA PCoE 数据集每块电池 `.mat` 含 charge / discharge / impedance 循环。  
SOH = 当前放电容量 / 首循环放电容量（对 discharge 电流积分得 Ah）。

任务设定：用过去 `W` 个循环的 SOH 序列，预测下一循环 SOH（无当前标签泄漏）。

## Week 2：RUL 预测 + 异常检测

```powershell
# RUL 对比实验（TCN / TCN-Attention / LSTM）
py scripts/train_rul.py --all --models tcn tcn-attn lstm --epochs 300

# 异常检测（TCN-Autoencoder）
py scripts/train_anomaly.py --all --epochs 200

# 一键跑对比 + 生成论文表格/图表
py scripts/run_week2_experiments.py --epochs-rul 300 --epochs-ae 200
```

### RUL 结果摘要（epochs=200）

| Battery | TCN RUL RMSE | TCN-Attn RUL RMSE | LSTM RUL RMSE |
|---------|--------------|-------------------|---------------|
| B0005   | 10.69        | **6.87**          | 57.32         |
| B0006   | **0.06**     | 0.23              | 18.23         |
| B0007   | **4.50**     | 5.61              | 57.43         |

B0005 上 TCN-Attention 相对 TCN 约降 36%；B0006/07 上并不总是更好，文中按电池分别报告。

### 异常检测结果（TCN-AE, p95 阈值）

| Battery | Accuracy | F1   |
|---------|----------|------|
| B0005   | 0.875    | 0.833 |
| B0006   | 0.750    | 0.714 |
| B0007   | 0.500    | 0.556 |

报告与图表：`outputs/week2/week2_report.md`、`rul_rmse_comparison.png`、`ablation_tcn_vs_attn.csv`

## 项目结构

```
tcn_attention/
├── data/nasa/
├── scripts/
│   ├── download_nasa.py
│   ├── train_soh_tcn.py      # Week1 SOH
│   ├── train_rul.py          # Week2 RUL Seq2Seq
│   ├── train_anomaly.py      # Week2 TCN-AE
│   ├── run_week2_experiments.py
│   ├── train_cross_dataset.py
│   └── run_enrichment_experiments.py  # 消融 + W/H/λ 敏感性
├── src/
│   ├── data/                 # nasa_loader, rul_dataset, cycle_features
│   ├── models/               # tcn, attention, rul_models, tcn_autoencoder
│   └── utils/
└── outputs/
    ├── rul/                  # RUL 模型与曲线
    ├── anomaly/              # 异常检测
    ├── week2/                # 对比报告
    ├── cross_dataset/        # NASA↔CALCE
    └── enrichment/           # 消融与敏感性
```

## 模块说明

| 模块 | 架构 | 关键机制 |
|------|------|----------|
| RUL  | TCN-Attention + Seq2Seq | TPA、Teacher Forcing、迭代多步校正 |
| 异常 | TCN-Autoencoder | 扩张因果卷积编码/解码、验证集 p95 阈值 |

## 下一步

- MC Dropout 置信区间
- 更多跨数据集实验
- 简单演示界面（可选）

## Week 3：EI 会议论文

- 主线：TCN-Attention 做 SOH/RUL；异常检测为辅，且仅用合成噪声验证
- RUL：B0005 改善明显，其他电池结果不一
- 数据：NASA + CALCE CS2（`outputs/cross_dataset/`）
- **改稿加厚（导师意见）**：组件消融 / \(W{,}H{,}\lambda\) 敏感性 / 参数量

```powershell
py scripts/train_cross_dataset.py --epochs-soh 100

# 论文最终实验（验证选模 + 三 seed；约 72 组）
py scripts/run_final_experiments.py --epochs 200
# 结果见 outputs/final/final_report.md
```

论文：`paper/ieee_conference.pdf`（中文对照 `paper/paper_draft_cn.md`；口径 `paper/SUBMISSION_NOTES.md`）
