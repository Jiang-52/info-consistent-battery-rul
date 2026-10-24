# Information-consistent battery RUL evaluation

Reproducibility package for the manuscript *Information-Consistent Evaluation of Li-ion Battery Remaining Useful Life*.

This repository provides the evaluation protocol (information levels L0 / L1 / L1P / L2; fit Protocols A and B), ranking instruments (MLP, Patch-TF), adopted experiment outputs, and scripts to regenerate the paper tables.

It is **not** an architecture leaderboard and does not claim a new predictive backbone.

## Repository layout

```
experiments/info_consistent/     # paper-facing experiment mounts (adopted runs)
outputs/                         # same adopted result JSON/MD (canonical on disk)
scripts/                         # reproduction entry points
src/                             # models and utilities
paper/info_consistent_manuscript/  # LaTeX manuscript + figures
tests/                           # unit checks (EOL / L1P causality)
```

### Adopted experiment outputs

| Path | Role |
|------|------|
| `outputs/info_levels_severson_loco/` | Severson Protocol B primary ladder (\(n=24\)) |
| `outputs/info_levels_v2/` | NASA + CALCE Protocol A ladder |
| `outputs/l1p_mechanism/` | L1P mechanism windows + dual-head SOH control |
| `outputs/revision_controls/` | Scale / support controls |
| `outputs/info_levels_severson/` | Severson Protocol A same-window check |

Index: `paper/info_consistent_manuscript/docs/EXPERIMENT_INDEX.md`.

## Setup

```powershell
cd <repo-root>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
# CPU torch is enough for the reported ranking runs
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Data (not shipped)

Place public cycle-life corpora under `data/` (gitignored):

- NASA PCoE B0005–B0007: `py scripts/download_nasa.py`
- CALCE CS2 cells used in Protocol A
- Severson / MATR batch files for Protocol B

Exact cell lists and \(T\) values for the scored Severson subset are in `outputs/info_levels_severson/selection.json`.

## Reproduce paper runs

Defaults match the manuscript (\(W{=}50\), \(H{=}5\), seeds `{42,43,44}`, 80 epochs unless noted).

```powershell
# NASA + CALCE, Protocol A (within-cell)
py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol within_cell --corpus nasa,calce

# Severson Protocol B (leave-target), stratified n=24
py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol leave_target --corpus severson --severson-n 24 --out-dir outputs/info_levels_severson_loco

# L1P mechanism + SOH-head / RUL÷T control (NASA Protocol A)
py scripts/run_l1p_mechanism.py --epochs 80

# Scale / channel-support controls
py scripts/run_revision_controls.py --epochs 40
```

Smoke checks (short epochs):

```powershell
py scripts/run_info_levels_v2.py --smoke --epochs 2
py -m pytest tests/ -q
```

After a full re-run, regenerate LaTeX tables if needed:

```powershell
py scripts/fill_info_paper_tables.py
py scripts/merge_severson_into_info_tables.py
```

Manuscript PDFs: `paper/info_consistent_manuscript/00_latest/`.

## Citation / availability

- GitHub: https://github.com/Jiang-52/info-consistent-battery-rul
- See `CODE_AVAILABILITY.md` for the journal Data Availability fields (Zenodo DOI pending).
