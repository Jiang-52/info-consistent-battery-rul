# 实验目录索引（投稿规范）

**论文采用实验已独立挂载：**  
`F:\科研工作区\tcn_attention\experiments\info_consistent\`  
（junction → `outputs/` 同名目录；详见该目录 `README.md` / `MANIFEST.json`）

根路径（原始落盘）：`F:\科研工作区\tcn_attention\outputs\`

## 采用（写进论文 / 填表）

1. **`info_levels_severson_loco/`** — Protocol B 主证据  
   - 中位梯子与 Wilcoxon：`summary.json`  
   - 表文：同步到 `generated_severson.tex`

2. **`info_levels_v2/`** — NASA + CALCE Protocol A  
   - 寿命表 / regen：`report.md`、`diagnostics.json`  
   - 阶梯 RMSE：`agg.json` → `generated_tables.tex`

3. **`l1p_mechanism/`** — Finding 2 机理  
   - `windows.json` 逐窗预测/通道/SOH/匹配距离；`summary.json` 分箱误差  
   - `soh_head.json` 梯度范数比与 **RUL/\(T\) 归一化损失** 对照（正文默认双头目标）

4. **`revision_controls/`** — Finding 2–3 封口对照  
   - `revision_controls_report.md`  
   - scale-control 为独立重跑，勿与 `tab:ladder` 混用 s.d.

5. **`info_levels_severson/`** — 同批细胞 Protocol A 对照窗

## 仅参考 / 勿替换主表

| 目录 | 原因 |
|---|---|
| `info_levels_l1p/` | 可能与主表 L1P 数字不一致；以 `generated_tables` 为准 |
| `info_levels_v2_rerun/` | 重跑草稿 |
| `*_smoke/` | 冒烟测试 |
| `info_levels/`（无后缀） | 早期协议 |
| `ei_v*` / `rul` / `week2` 等 | 旧 EI / 周报流水线 |

## 出图脚本

`paper/info_consistent_manuscript/make_fig_*.py` → `figures/`
