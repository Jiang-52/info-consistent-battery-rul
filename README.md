# info-consistent-battery-rul

Reproduce the information-level RUL protocol used in our evaluation study:

- levels: L0 (raw `t`), L1 (leave-target prior `t/T_prior`), L1P (SOH-matched sibling prior), L2 (`t/T_target`)
- fit: Protocol A (within-cell) / Protocol B (leave-target)

Models are MLP and Patch-TF under a shared training budget.

## Contents

| Path | What |
|------|------|
| `scripts/` | download + run scripts |
| `src/` | models / baselines / utils |
| `tests/` | small checks |
| `outputs/` | saved metrics for the reported runs |
| `experiments/info_consistent/` | same result folders (junctions / copies) |

Result folders:

- `outputs/info_levels_severson_loco` — Severson, Protocol B, n=24
- `outputs/info_levels_v2` — NASA + CALCE, Protocol A
- `outputs/l1p_mechanism` — per-window L1P/L2 diagnostics
- `outputs/revision_controls` — scale / support checks
- `outputs/info_levels_severson` — Severson, Protocol A

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Data (not included): put NASA / CALCE / Severson files under `data/`.

```powershell
py scripts/download_nasa.py
# Severson: py scripts/download_severson.py   # if you need the batch files
```

Cell list for the Severson subset: `outputs/info_levels_severson/selection.json`.

## Run

Default: window=50, horizon=5, seeds 42/43/44, epochs=80.

```powershell
py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol within_cell --corpus nasa,calce

py scripts/run_info_levels_v2.py --epochs 80 --fit-protocol leave_target --corpus severson --severson-n 24 --out-dir outputs/info_levels_severson_loco

py scripts/run_l1p_mechanism.py --epochs 80

py scripts/run_revision_controls.py --epochs 40
```

Smoke:

```powershell
py scripts/run_info_levels_v2.py --smoke --epochs 2
pytest tests/ -q
```
