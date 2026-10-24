# 信息一致 RUL 评价稿（info_consistent_manuscript）

> **先看最新：** [`00_latest/`](00_latest/)（PDF 终稿 + 实验路径一览）

评估方法学：$\operatorname{Rank}(\cdot\mid\mathcal{I}_t)=f(\mathcal{I}_t)$。  
不是 Patch-TF / Transformer SOTA，不是 PINN / 校准 UQ 主线。

---

## 目录结构（整理后）

```
info_consistent_manuscript/
├── 00_latest/              ← 最新 PDF 与交付说明（优先打开）
├── main.tex / main.pdf     ← 英文源稿与当前编译 PDF（正文）
├── sections_end.tex        ← 英文稿后半（讨论/结论/附录），由 main.tex 引入
├── main_zh.tex / main_zh.pdf
├── refs.tex                ← 参考文献（~39，2025–2026 过半）
├── generated_*.tex         ← 自动表（正文 ladder/persist；附录 soh/stats；Severson；clock；UQ）
├── figures/                ← 投稿图 PDF/PNG
├── make_fig_*.py           ← 出图脚本
├── docs/                   ← 修订说明、导师意见对照
├── _build/                 ← LaTeX 中间文件（可删）
├── _archive/               ← 旧 PDF 副本
└── _texbin/                ← 便携 tectonic
```

---

## 最新论文

| 语言 | 打开 |
|---|---|
| 英文 | [`00_latest/EN_main.pdf`](00_latest/EN_main.pdf) 或根目录 `main.pdf` |
| 中文 | [`00_latest/ZH_main.pdf`](00_latest/ZH_main.pdf) 或根目录 `main_zh.pdf` |

配套：`docs/导师意见修改对照与回复.txt`，`docs/CHANGELOG_REVISION.md`。

---

## 最新实验（规范 / 投稿用）

项目根：`F:\科研工作区\tcn_attention\`  
**独立挂载（投稿用）：** [`experiments/info_consistent/`](../../experiments/info_consistent/)

| 角色 | 目录 | 关键文件 |
|---|---|---|
| **主证据 Protocol B** | `outputs/info_levels_severson_loco/` | `summary.json`, `report.md`, `agg.json` |
| **NASA/CALCE Protocol A** | `outputs/info_levels_v2/` | `report.md`, `agg.json`, `diagnostics.json` |
| **L1P 机理 + 归一化损失** | `outputs/l1p_mechanism/` | `windows.json`, `soh_head.json` |
| **Priority-1 对照** | `outputs/revision_controls/` | `revision_controls_report.md` |
| Severson Protocol A 对照 | `outputs/info_levels_severson/` | `summary.json` |

**非主结果（勿当作投稿表）：** `*_smoke/`、`info_levels_v2_rerun/`、早期 `outputs/info_levels/`、`ei_v*`、`info_levels_l1p/`。

索引：`docs/EXPERIMENT_INDEX.md`。

### 复现命令

```powershell
cd F:\研究生项目\tcn_attention
.\.venv\Scripts\python.exe scripts\run_info_levels_v2.py --epochs 80
.\.venv\Scripts\python.exe scripts\run_revision_controls.py
# Severson leave-target（以仓库现有脚本/参数为准）
.\.venv\Scripts\python.exe paper\info_consistent_manuscript\make_fig_severson_b.py
.\.venv\Scripts\python.exe paper\info_consistent_manuscript\make_fig_l1p_nasa.py
```

---

## 编译

```powershell
cd F:\研究生项目\tcn_attention\paper\info_consistent_manuscript
.\_texbin\tectonic.exe --keep-logs --outdir . main.tex
.\_texbin\tectonic.exe --keep-logs --outdir . main_zh.tex
# 同步到 latest
Copy-Item -Force main.pdf 00_latest\EN_main.pdf
Copy-Item -Force main_zh.pdf 00_latest\ZH_main.pdf
```

---

## 主结果速查

- Severson B 中位 RMSE：L0 222.1 → L1 33.5 → L1P 30.3 → L2 12.0  
- L0>L2：24/24；Wilcoxon $p=1.2\times10^{-7}$  
- NASA A：L1P 可优于 L2（B0005：5.06 vs 23.58）