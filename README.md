# Information-consistent battery RUL evaluation

Code and adopted experiment outputs for reproducing the information-level RUL ranking protocol (L0 / L1 / L1P / L2; fit Protocols A and B) with matched MLP and Patch-TF instruments.

## Layout

```
scripts/                      # reproduction entry points
src/                          # models and utilities
tests/                        # small unit checks
outputs/                      # adopted result JSON / MD (see table)
experiments/info_consistent/  # same adopted runs (convenience mounts)
```

| Path | Role |
|------|------|
| `outputs/info_levels_severson_loco/` | Severson Protocol B ladder (\(n=24\)) |
| `outputs/info_levels_v2/` | NASA + CALCE Protocol A ladder |
| `outputs/l1p_mechanism/` | L1P mechanism windows + dual-head SOH control |
| `outputs/revision_controls/` | Scale / support controls |
| `outputs/info_levels_severson/` | Severson Protocol A same-window check |

## Setup

```powershell
cd <repo-root>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Data (not in this repo)

Put public corpora under `data/` (gitignored):

- NASA PCoE B0005–B0007: `py scripts/download_nasa.py`
- CALCE CS2 cells used in Protocol A
- Severson / MATR files for Protocol B

Scored Severson cell ids and lifetimes: `outputs/info_levels_severson/selection.json`.

## Reproduce

Defaults: \(W{=}50\), \(H{=}5\), seeds `{42,43,44}`, 80 epochs unless noted.

```powershell
# NASA + CALCE, Protocol A
py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol within_cell --corpus nasa,calce

# Severson Protocol B, stratified n=24
py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol leave_target --corpus severson --severson-n 24 --out-dir outputs/info_levels_severson_loco

# L1P mechanism + SOH-head control (NASA Protocol A)
py scripts/run_l1p_mechanism.py --epochs 80

# Scale / channel-support controls
py scripts/run_revision_controls.py --epochs 40
```

Quick checks:

```powershell
py scripts/run_info_levels_v2.py --smoke --epochs 2
py -m pytest tests/ -q
```

Compare regenerated metrics against the JSON/MD already under `outputs/`.
