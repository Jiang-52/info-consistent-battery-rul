# 最新交付物（看这里）

更新时间：2026-10-24（老师摘要/引言 + 健康双头默认 + 实验独立 + git）

## 论文（当前终稿）

| 文件 | 说明 |
|---|---|
| `EN_main.pdf` | 英文投稿稿（稳定文件名） |
| `ZH_main.pdf` | 中文对照稿（稳定文件名） |

源文件：上级 `main.tex` + `sections_end.tex` / `main_zh.tex` / `refs.tex`。

**定位：** RUL 信息一致评价方法学（\(\operatorname{Rank}(\cdot\mid\mathcal{I}_t)=f(\mathcal{I}_t)\)），不是新架构。

**主证据：** Severson Protocol B，\(n=24\)，中位 RMSE \(222.1\to12.0\)；Wilcoxon \(p=1.2\times10^{-7}\)。

**本轮：** 摘要按老师表述替换；引言按 背景→痛点→缺口→方案→发现→贡献；双头默认 \(\mathcal{L}_{\mathrm{SOH}}+0.5\mathcal{L}_{\mathrm{RUL}/T}\)。

## 实验

独立目录：`F:\科研工作区\tcn_attention\experiments\info_consistent\`  
详见 `../docs/EXPERIMENT_INDEX.md`。

## 一键编译

```powershell
cd F:\科研工作区\tcn_attention\paper\info_consistent_manuscript
.\_texbin\tectonic.exe -X compile --outdir _build main.tex
.\_texbin\tectonic.exe -X compile --outdir _build main_zh.tex
Copy-Item _build\main.pdf 00_latest\EN_main.pdf
Copy-Item _build\main_zh.pdf 00_latest\ZH_main.pdf
```

## 代码 / 数据可用性（投稿栏）

- 本地 git：`F:\科研工作区\tcn_attention`（已 init）
- GitHub 公开仓库 URL / Zenodo DOI：首次 `git push` 或 Zenodo deposit 后填入此处与 Data Availability。
