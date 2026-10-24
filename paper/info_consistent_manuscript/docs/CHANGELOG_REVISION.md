# Structural revision changelog (mentor brief)

## 2026-10-24 — mentor abstract / intro / healthy dual-head / experiments+git

- **Abstract (EN/ZH):** replaced with mentor-aligned wording (information boundary; nested L0–L2/L1P; Protocols A/B; Severson B \(222.1\to12.0\); NASA L1P>L2 via out-of-support; methods standard not new arch).
- **Introduction:** rewritten as background → pain → three gaps → framework → two findings → C1–C3.
- **Dual-head:** default objective set to \(\mathcal{L}_{\mathrm{SOH}}+0.5\mathcal{L}_{\mathrm{RUL}/T}\); Table `sohcontrol` reframed as health check + healthy-model RUL supplement (hierarchy unchanged).
- **Experiments isolated:** `experiments/info_consistent/` (junctions to adopted `outputs/`); `docs/EXPERIMENT_INDEX.md` updated.
- **Git:** local repo at `F:\科研工作区\tcn_attention` (GitHub/Zenodo remote URL to be added after first push / deposit).
- **PDFs:** rebuilt to `00_latest/EN_main.pdf` and `00_latest/ZH_main.pdf`.

## Positioning (unchanged)

- Methods / evaluation paper: $\operatorname{Rank}(\cdot\mid\mathcal{I}_t)=f(\mathcal{I}_t)$.
- Not a Patch-TF / Transformer SOTA claim; MLP and Patch-TF are ranking instruments only.
- Not a PINN zoo; not a calibrated UQ / conformal spine.

## Title / abstract

| Old | New |
|---|---|
| Nested Permitted Lifetime Constants and a Deliberate Oracle | A Nested Information-Level Framework for Retrospective Oracles |
| Long abstract listing every protocol detail | Rank question + Findings 1–2 + demoted affine claim + methods contribution |

## Old → new conclusions

| Topic | Old claim | New claim |
|---|---|---|
| Affine Eq. | Model-facing: L1≈L2 expected under scale invariance | Input-level only; L1≈L2 is empirical scale robustness (controls), not invariance theorem |
| L1P | Minor non-affine diagnostic | Elevated carefully: can undershoot L2 on NASA Protocol A (Finding 2); matching distances reported |
| Primary evidence | NASA ladder + Severson both prominent | Severson Protocol B ($n=24$) is primary (Finding 1); NASA/CALCE exploratory |
| Protocol A L0 vs L2 | Soft mention that L0 can be easier | Explicit Finding 3 + scale/range controls (L0 clock $t/1000$, 24% test extrapolation on B0005) |
| SOH dual-head | Auxiliary SOH RMSE table | Dual-head vs SOH-only: multi-task interference (0.02–0.04 → 0.56–0.81); not an SOH estimation claim |
| UQ | Appendix MC-Dropout | Demoted; conformal/PINN out of scope |

## Manuscript structure edits (`main.tex` / `main_zh.tex`)

- Intro: three gaps; Findings 1–2; contributions C1–C5.
- Protocol: decision guide A vs B; formal L1P median-nearest definition; affine downgrade.
- Results: Findings 1–3 lead; Severson B figure; L1P NASA figure; §Revision controls.
- Discussion: Findings 1–3; “what this paper is not”; limitations.

## New figures

- `figures/fig_severson_protocol_b.pdf` — primary Severson Protocol B ladder.
- `figures/fig_l1p_nasa.pdf` — NASA L1P vs L0/L1/L2.

## Priority-1 controls (completed)

Script: `scripts/run_revision_controls.py` → `outputs/revision_controls/`.

| Control | Result (summary) |
|---|---|
| L1P matching distance | B0005/6/7 mean SOH dist 0.0039 / 0.0307 / 0.0010 |
| Scale control (MLP) | L1≈L2; L0 often lower under Protocol A; $t/1000$ ≈ L1/L2 on B0005 |
| L1 vs L2 pred Δ | mean \|Δ\| ≪ 1 cycle; corr > 0.97 |
| SOH-only vs dual | SOH RMSE 0.02–0.04 vs 0.56–0.81 at L2 |
| Protocol A ranges | B0005: 24% test windows above train-max cycle channel |

## Experiment list (canonical)

1. `scripts/run_info_levels_v2.py` — NASA/CALCE Protocol A ladder (L0/L1/L1P/L2).
2. Severson Protocol A / B ladders — `outputs/info_levels_severson/`, `outputs/info_levels_severson_loco/`.
3. `scripts/run_revision_controls.py` — scale, matching, SOH heads, ranges.
4. Figure makers: `make_fig_severson_b.py`, `make_fig_l1p_nasa.py`, `make_fig_info_levels*.py`, `make_fig_soh_trajectories.py`.

## Stats hierarchy

1. **Primary:** Severson Protocol B medians 222.1 / 33.5 / 30.3 / 12.0; Wilcoxon $p=1.2\times10^{-7}$.
2. **Secondary:** NASA L1P < L2 (e.g. B0005 MLP 5.06 vs 23.58).
3. **Diagnostic:** affine / scale / ranges / SOH heads.
4. **Exploratory:** NASA+CALCE $n=7$ ($p=0.219$); CALCE Patch-TF not an architecture ranking.

## Finalization pass (post mentor-alignment review)

| Item | Action |
|---|---|
| Table `tab:clock` OLS | Moved to **Appendix**; body keeps one identity-check sentence |
| Leakage wording | Prefer retrospective oracle / online information-inconsistent; leakage reserved for use-case illegitimacy |
| Contributions | C1–C4 only; reporting template demoted to practical add-on |
| Abstract / Intro | Online condition on L2; Most→Many; rarely→often do not explicitly; Rank as shorthand |
| Scale control | Explicit: L1/L2 partly numerical scale, not only semantic $T$ |
| References | High-density set: **29** bibitems; **9** from 2024–2026 |
| Number audit | Abstract/Intro/Severson tables/L1P NASA–CALCE figures reconciled (222.1/33.5/30.3/12.0; 5.06 vs 23.58; 60.7 vs 2.20; p=1.2e-7 / 0.219) |

## Nature-skills pre-submission pass

| Item | Action |
|---|---|
| Results order | Severson Protocol B moved before NASA ladder / legacy (primary-first narrative) |
| Scale-control labeling | Independent re-run; do not mix s.d./L0 with `tab:ladder` |
| Seed Wilcoxon | Descriptive only; primary inference = Severson B cell-level |
| Self-ref | Abstract/Discussion: This paper |
| ZH sync | `main_zh.tex` matched on order + scale-control + seed wording |

## Literature + prose refresh (2026-10-16)

| Item | Action |
|---|---|
| References | Expanded to **39** bibitems; **20** from 2025–2026 (majority); Crossref-checked DOIs |
| EN abstract | Problem → oracle → protocol design → primary result → boundary → contribution |
| Intro / Related work | Clearer funnel; cite LOCO/group-wise and recent RUL models without architecture claim |
| ZH abstract / 相关工作 | Aligned with EN logic; removed broken old cite keys |

## Final technical seal (mentor re-read)

| Item | Action |
|---|---|
| Protocol A train/val | Chronological early 80% support; within support, last 15% val / 85% train; scaler on train only; early-stop on val RUL RMSE; overlap caveat |
| Protocol B val | Seed-fixed sibling shuffle; last 15% val |
| Title | Drop “Nested”; EN/ZH → information-level framework |
| Number checksum | Table 2 regen/EOL match `outputs/info_levels_v2/report.md`; Severson medians 222.1/33.5/30.3/12.0 + Wilcoxon \(p=1.19\times10^{-7}\) match `info_levels_severson_loco/summary.json`; NASA L1P 5.06 and CS2_36 60.69 match `generated_tables.tex` (ignore stale `info_levels_l1p/agg.json`) |

## Deliverable map

| Item | Path |
|---|---|
| EN manuscript | `main.tex` → `main.pdf` (rebuilt after finalization) |
| ZH narrative | `main_zh.tex` → `main_zh.pdf` |
| Tables | `generated_tables.tex`, `generated_severson.tex`, `generated_uq.tex` |
| Refs | `refs.tex` (~36 bibitems; 2024–2026 elevated) |
| This changelog | `CHANGELOG_REVISION.md` |

## Major revision M1--M7 (2026-10-23)

| Item | Action |
|---|---|
| M1 L1P mechanism | New §5.3 + Fig. 3 + Table 9; support-extrapolation mechanism; boundary under B/CALCE |
| M2 SOH dual-head | New Table 13; gradient ratio 64--130; loss-normalised repair; full table moved to Appendix |
| M3 affine logic | New Table 1 (proved / implied / not implied); empirical scale vs invariance clarified |
| M4 A/B guide | §3.3 expanded + Table 2 (single-cell, homogeneous, new-target, online cases) |
| M5 censored T | T as piecewise function; “EOL reached” column in Table 4; B0007 caveat in §5.7 |
| M6 statistics | Primary inference restricted to Severson B; NASA/CALCE relabelled exploratory; seed table to Appendix |
| M7 platforms | New Related-work and Discussion paragraphs on BatteryML / BatteryLife |
| Contributions | Reorganised into C1 core / C2 mechanism / C3 reporting; L1P elevated |
| Abstract | Four-element definition + L1P mechanism sentence + protocol-B reversal |
| New artefacts | scripts/run_l1p_mechanism.py; outputs/l1p_mechanism/; fig_l1p_mechanism.pdf; generated_appendix.tex; sections_end.tex |
| Split | main.tex + sections_end.tex (write-size limit); body/appendix tables split |
| Response | docs/REVISION_RESPONSE_M1-M7.md |
