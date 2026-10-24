#!/usr/bin/env python3
"""Information-consistent RUL/SOH evaluation, advisor-revision protocol.

Lifetime T := first cycle with SOH <= 0.7 (EOL), not last recorded cycle.
That definition makes NASA L1 != L2 (EOL = 106/161/167 for B0006/5/7).

L2: cycle_channel = t / T_target
L1: cycle_channel = t / median(T_siblings)  (leave-target-out prior)
L0: cycle_channel = raw t  (no lifetime encoding; optional L0_clock = t/1000)

Fit protocols:
  within_cell  — chronological 80/20 on the target cell (Protocol A).
                 Training labels RUL=T_target-t encode that cell's lifetime.
  leave_target — train/val on sibling cells only; same target late-20% test
                 windows (Protocol B). T_target never enters fitting.

Persistence variants:
  persist_fade  — constant SOH fade in the window (legacy "persistence")
  persist_hold  — RUL_hat_t = RUL_{t-1}
  persist_mean  — expanding mean of historical RUL
  linear        — least-squares SOH fit in the window

Outputs: outputs/info_levels_v2/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_rul import set_seed, _eval_rul_rmse
from src.baselines.classical import evaluate_classical
from src.data.calce_loader import list_calce_cells, load_calce_cell
from src.data.cycle_features import (
    INFO_L0,
    INFO_L1,
    INFO_L1P,
    INFO_L2,
    apply_cycle_info_level,
    apply_soh_matched_rul_prior,
    extract_discharge_cycle_features,
)
from src.data.nasa_loader import chronological_split, load_battery_soh
from src.data.rul_dataset import (
    build_multifeat_rul_windows,
    eol_cycle_index,
    load_battery_rul_data,
)
from src.data.severson_loader import (
    EOL_FRAC as SEVERSON_EOL_FRAC,
    list_severson_cells,
    load_compact as load_severson_compact,
    load_severson_cell,
    save_compact as save_severson_compact,
)
from src.models.rul_models import build_rul_model
from src.models.uncertainty import mc_dropout_predict, picp, pinaw, physics_regularizers
from src.utils.metrics import regression_metrics

SEEDS = [42, 43, 44]
NASA_CELLS = ["B0005", "B0006", "B0007"]
EOL_TH = 0.7
SOH_CACHE: dict[tuple[str, str], np.ndarray] = {}


def cohens_d(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp < 1e-12:
        return 0.0
    return float((x.mean() - y.mean()) / sp)


def wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return None
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y) or len(x) < 2:
        return None
    d = x - y
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
    except Exception:
        return None


def soh_series_diagnostics(soh: np.ndarray, cell_id: str, corpus: str) -> dict:
    soh = np.asarray(soh, dtype=np.float64).reshape(-1)
    d = np.diff(soh)
    eol = int(eol_cycle_index(soh, EOL_TH))
    return {
        "corpus": corpus,
        "cell": cell_id,
        "n_cycles": int(len(soh)),
        "eol70": eol,
        "soh_start": float(soh[0]),
        "soh_end": float(soh[-1]),
        "soh_min": float(soh.min()),
        "regen_sum": float(np.sum(d[d > 0])),
        "n_up_steps": int(np.sum(d > 1e-4)),
        "std_delta": float(np.std(d)),
        "mean_fade": float(-np.mean(d)),
        "late_mean_delta": float(np.mean(d[int(0.8 * len(d)) :])) if len(d) > 5 else float("nan"),
    }


def nasa_lifetime_table(data_dir: Path) -> dict[str, dict]:
    out = {}
    mats = sorted(data_dir.glob("B*.mat"))
    ids = [p.stem.upper() for p in mats] or list(NASA_CELLS)
    for bat in ids:
        path = data_dir / f"{bat}.mat"
        if not path.exists():
            continue
        _, _, soh = load_battery_soh(path, bat)
        SOH_CACHE[("nasa", bat)] = np.asarray(soh, dtype=np.float32).reshape(-1)
        raw = extract_discharge_cycle_features(path, bat, normalize_cycle=False)
        d = soh_series_diagnostics(soh, bat, "nasa")
        d["t_record"] = float(raw[-1, 5])
        out[bat] = d
    return out


def calce_lifetime_table(npy: Path) -> dict[str, dict]:
    out = {}
    for cid in list_calce_cells(npy):
        _, _, soh = load_calce_cell(npy, cid)
        SOH_CACHE[("calce", cid)] = np.asarray(soh, dtype=np.float32).reshape(-1)
        d = soh_series_diagnostics(soh, cid, "calce")
        d["t_record"] = float(len(soh) - 1)
        out[cid] = d
    return out


def severson_lifetime_table(compact_json: Path) -> dict[str, dict]:
    cells = load_severson_compact(compact_json)
    out = {}
    for cid, cell in cells.items():
        soh = np.asarray(cell["soh"], dtype=np.float32).reshape(-1)
        SOH_CACHE[("severson", cid)] = soh
        d = soh_series_diagnostics(soh, cid, "severson")
        # Override EOL with published Severson 80%-of-nominal definition.
        d["eol70"] = int(cell["eol80"])  # field name kept for sibling_t_eol
        d["eol80"] = int(cell["eol80"])
        d["t_record"] = float(cell["t_record"])
        d["eol_threshold"] = float(SEVERSON_EOL_FRAC)
        out[cid] = d
    return out


def load_severson_rul_windows(
    compact_json: Path,
    cell_id: str,
    window: int,
    horizon: int,
    info_level: str,
    t_scale: float,
    eol_threshold: float = SEVERSON_EOL_FRAC,
    l1p_exclude: set[str] | None = None,
) -> dict:
    _, _, soh = load_severson_cell(compact_json, cell_id)
    soh = np.asarray(soh, dtype=np.float32).reshape(-1)
    t = np.arange(len(soh), dtype=np.float32)
    raw = np.stack(
        [soh, np.zeros_like(soh), np.zeros_like(soh), np.zeros_like(soh), np.zeros_like(soh), t],
        axis=1,
    )
    level = info_level.upper()
    if level == INFO_L1P:
        feats = apply_soh_matched_rul_prior(
            raw, soh, _sibling_sohs("severson", cell_id, l1p_exclude), eol_threshold
        )
        level_store = INFO_L1P
    elif level == "L0_CLOCK":
        feats = apply_cycle_info_level(raw, INFO_L0, t_scale=t_scale)
        level_store = "L0_CLOCK"
    else:
        feats = apply_cycle_info_level(raw, level, t_scale=t_scale)
        level_store = level
    x, y_soh, y_rul = build_multifeat_rul_windows(feats, soh, window, horizon, eol_threshold)
    return {
        "x": x,
        "y_soh": y_soh,
        "y_rul": y_rul,
        "soh": soh,
        "target_last_idx": float(t[-1]),
        "eol_cycle": int(eol_cycle_index(soh, eol_threshold)),
        "info_level": level_store,
        "t_scale": float(t_scale),
        "n_features": int(x.shape[-1]),
    }


def stratify_severson_cells(life: dict[str, dict], n: int) -> list[str]:
    """Pick short/mid/long EOL cells for a manageable heterogeneous ladder."""
    items = sorted(life.items(), key=lambda kv: float(kv[1]["eol70"]))
    if n >= len(items):
        return [k for k, _ in items]
    n = max(3, n)
    n_each = n // 3
    short = items[:n_each]
    long = items[-n_each:]
    mid_start = max(0, (len(items) - n_each) // 2)
    mid = items[mid_start : mid_start + n_each]
    picked = []
    for block in (short, mid, long):
        for k, _ in block:
            if k not in picked:
                picked.append(k)
    # fill remainder from unused middle
    if len(picked) < n:
        for k, _ in items:
            if k not in picked:
                picked.append(k)
            if len(picked) >= n:
                break
    return picked[:n]


def sibling_t_eol(
    table: dict[str, dict], target: str, also_exclude: set[str] | None = None
) -> dict:
    """T_prior from leave-target training cells only — never median(T_all)."""
    ban = {target} | (also_exclude or set())
    others = [float(table[k]["eol70"]) for k in table if k not in ban]
    if not others:
        raise ValueError(f"Need at least one sibling for L1 prior ({target})")
    all_t = [float(table[k]["eol70"]) for k in table]
    return {
        "siblings": {k: float(table[k]["eol70"]) for k in table if k not in ban},
        "t_med": float(np.median(others)),
        "t_med_all_illegal": float(np.median(all_t)),
        "t_min": float(np.min(others)),
        "t_max": float(np.max(others)),
        "t_target": float(table[target]["eol70"]),
        "t_record": float(table[target]["t_record"]),
        "used_target_in_prior": False,
    }


def t_scale_for(level: str, scales: dict, clock: float = 1000.0) -> float:
    level = level.upper()
    if level == INFO_L2:
        return float(scales["t_target"])
    if level == INFO_L1:
        return float(scales["t_med"])
    if level == INFO_L1P:
        return float(scales["t_med"])
    if level == INFO_L0:
        return 1.0
    if level in {"L0_CLOCK", "L0C"}:
        return float(clock)
    raise ValueError(level)


def _sibling_sohs(corpus: str, cell_id: str, also_exclude: set[str] | None = None) -> list[np.ndarray]:
    ban = {cell_id} | (also_exclude or set())
    out = [v for (c, k), v in SOH_CACHE.items() if c == corpus and k not in ban]
    if not out:
        raise ValueError(f"L1P needs sibling SOH series for {corpus}:{cell_id}")
    return out


def load_calce_rul_windows(
    npy: Path,
    cell_id: str,
    window: int,
    horizon: int,
    info_level: str,
    t_scale: float,
    l1p_exclude: set[str] | None = None,
) -> dict:
    _, cap, soh = load_calce_cell(npy, cell_id)
    soh = np.asarray(soh, dtype=np.float32).reshape(-1)
    t = np.arange(len(soh), dtype=np.float32)
    raw = np.stack(
        [soh, np.zeros_like(soh), np.zeros_like(soh), np.zeros_like(soh), np.zeros_like(soh), t],
        axis=1,
    )
    level = info_level.upper()
    if level == INFO_L1P:
        feats = apply_soh_matched_rul_prior(
            raw, soh, _sibling_sohs("calce", cell_id, l1p_exclude), EOL_TH
        )
        level_store = INFO_L1P
    elif level == "L0_CLOCK":
        feats = apply_cycle_info_level(raw, INFO_L0, t_scale=t_scale)
        level_store = "L0_CLOCK"
    else:
        feats = apply_cycle_info_level(raw, level, t_scale=t_scale)
        level_store = level
    x, y_soh, y_rul = build_multifeat_rul_windows(feats, soh, window, horizon, EOL_TH)
    return {
        "x": x,
        "y_soh": y_soh,
        "y_rul": y_rul,
        "soh": soh,
        "target_last_idx": float(t[-1]),
        "eol_cycle": int(eol_cycle_index(soh, EOL_TH)),
        "info_level": level_store,
        "t_scale": float(t_scale),
        "n_features": int(x.shape[-1]),
    }


def restrict_pre_eol(data: dict, min_windows: int = 24) -> dict:
    """Keep windows with prediction time before first EOL (RUL_t = max(0, T - t) > 0)."""
    y = np.asarray(data["y_rul"]).reshape(-1)
    mask = y > 0
    n = int(mask.sum())
    if n < min_windows:
        raise ValueError(f"Too few pre-EOL windows ({n}); cannot split.")
    out = dict(data)
    out["x"] = data["x"][mask]
    out["y_soh"] = data["y_soh"][mask]
    out["y_rul"] = data["y_rul"][mask]
    out["n_pre_eol"] = n
    out["n_dropped_post_eol"] = int((~mask).sum())
    return out


def load_nasa_windows(
    data_dir: Path,
    battery: str,
    window: int,
    horizon: int,
    info_level: str,
    t_scale: float,
    l1p_exclude: set[str] | None = None,
) -> dict:
    level = info_level.upper()
    if level == INFO_L1P:
        _, _, soh = load_battery_soh(data_dir / f"{battery}.mat", battery)
        raw = extract_discharge_cycle_features(
            data_dir / f"{battery}.mat", battery, normalize_cycle=False
        )
        n = min(len(raw), len(soh))
        raw, soh = raw[:n], soh[:n]
        feats = apply_soh_matched_rul_prior(
            raw, soh, _sibling_sohs("nasa", battery, l1p_exclude), EOL_TH
        )
        x, y_soh, y_rul = build_multifeat_rul_windows(feats, soh, window, horizon, EOL_TH)
        return {
            "x": x,
            "y_soh": y_soh,
            "y_rul": y_rul,
            "soh": soh,
            "eol_cycle": int(eol_cycle_index(soh, EOL_TH)),
            "info_level": INFO_L1P,
            "t_scale": float(t_scale),
            "n_features": int(x.shape[-1]),
        }
    store = level
    use_level = INFO_L0 if level == "L0_CLOCK" else level
    data = load_battery_rul_data(
        data_dir / f"{battery}.mat",
        battery,
        window,
        horizon,
        feature_mode="multifeat",
        info_level=use_level,
        t_scale=t_scale,
        eol_threshold=EOL_TH,
    )
    data["info_level"] = store
    return data


def load_cell_windows(
    corpus: str,
    cell_id: str,
    window: int,
    horizon: int,
    info_level: str,
    t_scale: float,
    data_dir: Path,
    npy: Path,
    sev_json: Path,
    l1p_exclude: set[str] | None = None,
) -> dict:
    if corpus == "nasa":
        return load_nasa_windows(
            data_dir, cell_id, window, horizon, info_level, t_scale, l1p_exclude=l1p_exclude
        )
    if corpus == "calce":
        return load_calce_rul_windows(
            npy, cell_id, window, horizon, info_level, t_scale, l1p_exclude=l1p_exclude
        )
    if corpus == "severson":
        return load_severson_rul_windows(
            sev_json, cell_id, window, horizon, info_level, t_scale, l1p_exclude=l1p_exclude
        )
    raise ValueError(corpus)


def pool_sibling_windows(
    corpus: str,
    target: str,
    sibling_ids: list[str],
    window: int,
    horizon: int,
    info_level: str,
    life_table: dict[str, dict],
    data_dir: Path,
    npy: Path,
    sev_json: Path,
    max_windows: int = 20000,
    seed: int = 42,
) -> dict:
    """Concatenate pre-EOL windows from leave-target siblings (Protocol B train pool)."""
    xs, ys, yr = [], [], []
    # Scored target never enters L1P sibling matching during fitting.
    ban_target = {target}
    for cid in sibling_ids:
        if cid == target or cid not in life_table:
            continue
        # Exclude scored target from every train-sibling prior / L1P map.
        scales = sibling_t_eol(life_table, cid, also_exclude={target})
        # For L2 on a train sibling, use that sibling's own T (available in D_train).
        # Override: sibling_t_eol stores cid's T in t_target field.
        ts = t_scale_for(info_level, scales)
        try:
            d = load_cell_windows(
                corpus,
                cid,
                window,
                horizon,
                info_level,
                ts,
                data_dir,
                npy,
                sev_json,
                l1p_exclude=ban_target,
            )
            d = restrict_pre_eol(d, min_windows=8)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"  skip sibling {cid}: {exc}", flush=True)
            continue
        xs.append(d["x"])
        ys.append(d["y_soh"])
        yr.append(d["y_rul"])
    if not xs:
        raise ValueError(f"No sibling windows for leave-target train ({corpus}:{target})")
    x = np.concatenate(xs, axis=0)
    y_soh = np.concatenate(ys, axis=0)
    y_rul = np.concatenate(yr, axis=0)
    if len(x) > max_windows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=max_windows, replace=False))
        x, y_soh, y_rul = x[idx], y_soh[idx], y_rul[idx]
    return {"x": x, "y_soh": y_soh, "y_rul": y_rul, "n_pool_windows": int(len(x))}


def train_one(
    model_name: str,
    cell_id: str,
    info_level: str,
    data: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    window: int,
    horizon: int,
    device: torch.device,
    corpus: str,
    t_scale: float,
    patch_size: int = 5,
    n_mc: int = 20,
    fit_protocol: str = "within_cell",
    train_pool: dict | None = None,
) -> dict:
    set_seed(seed)
    x, y_soh, y_rul = data["x"], data["y_soh"], data["y_rul"]
    # Same target late-20% test windows for both protocols (comparable RMSE).
    x_tr_full, x_te, y_soh_tr_full, y_soh_te = chronological_split(x, y_soh, 0.8)
    y_rul_tr_full = y_rul[: len(x_tr_full)]
    y_rul_te = y_rul[len(x_tr_full) :]
    fit_protocol = fit_protocol.lower().strip()
    if fit_protocol == "leave_target":
        if train_pool is None:
            raise ValueError("leave_target requires train_pool")
        # Target-cell early windows are discarded from fitting.
        x_tr_full = train_pool["x"]
        y_soh_tr_full = train_pool["y_soh"]
        y_rul_tr_full = train_pool["y_rul"]
        # Shuffle then take last 15% as val (siblings are mixed cells).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x_tr_full))
        x_tr_full = x_tr_full[perm]
        y_soh_tr_full = y_soh_tr_full[perm]
        y_rul_tr_full = y_rul_tr_full[perm]
    elif fit_protocol != "within_cell":
        raise ValueError(fit_protocol)
    n = len(x_tr_full)
    n_val = max(1, int(round(n * 0.15)))
    n_tr = max(1, n - n_val)
    x_tr, x_val = x_tr_full[:n_tr], x_tr_full[n_tr:]
    y_soh_tr, y_soh_val = y_soh_tr_full[:n_tr], y_soh_tr_full[n_tr:]
    y_rul_tr, y_rul_val = y_rul_tr_full[:n_tr], y_rul_tr_full[n_tr:]

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_tr), torch.tensor(y_soh_tr), torch.tensor(y_rul_tr)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_soh_val), torch.tensor(y_rul_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_te), torch.tensor(y_soh_te), torch.tensor(y_rul_te)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_rul_model(
        model_name,
        horizon=horizon,
        n_features=int(x.shape[-1]),
        window_size=window,
        patch_size=patch_size,
        d_model=64,
        num_layers=2,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    soh_fn, rul_fn = nn.MSELoss(), nn.MSELoss()
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb_soh, yb_rul in train_loader:
            xb, yb_soh, yb_rul = xb.to(device), yb_soh.to(device), yb_rul.to(device)
            opt.zero_grad()
            ps, pr = model(xb, yb_soh, 0.0)
            loss = soh_fn(ps, yb_soh) + 0.5 * rul_fn(pr, yb_rul) + physics_regularizers(ps)
            loss.backward()
            opt.step()
        val = _eval_rul_rmse(model, val_loader, device)
        if val < best_val:
            best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(
                f"  [{corpus}|{model_name}|{cell_id}|{info_level}|s{seed}] "
                f"ep {epoch}/{epochs} val={val:.3f} best@{best_epoch}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    rul_p, rul_t, soh_p, soh_t = [], [], [], []
    with torch.no_grad():
        for xb, yb_soh, yb_rul in test_loader:
            ps, pr = model(xb.to(device), None, 0.0)
            rul_p.append(pr.cpu().numpy())
            rul_t.append(yb_rul.numpy())
            soh_p.append(ps.cpu().numpy())
            soh_t.append(yb_soh.numpy())
    rp = np.concatenate(rul_p).reshape(-1)
    rt = np.concatenate(rul_t).reshape(-1)
    sp = np.concatenate(soh_p).reshape(-1)
    st = np.concatenate(soh_t).reshape(-1)
    mr = regression_metrics(rt, rp)
    ms = regression_metrics(st, sp)

    xb_te = torch.tensor(x_te)
    mc = mc_dropout_predict(model, xb_te, n_samples=n_mc, device=device)
    uq = {
        "picp_rul": picp(rt, mc["rul_mean"], mc["rul_std"]),
        "pinaw_rul": pinaw(rt, mc["rul_mean"], mc["rul_std"]),
        "picp_soh": picp(st, mc["soh_mean"], mc["soh_std"]),
        "pinaw_soh": pinaw(st, mc["soh_mean"], mc["soh_std"]),
        "n_mc": n_mc,
    }

    return {
        "experiment": "info_levels_v2",
        "corpus": corpus,
        "model": model_name,
        "battery": cell_id,
        "info_level": info_level,
        "fit_protocol": fit_protocol,
        "t_scale": float(t_scale),
        "eol70": int(data.get("eol_cycle", -1)),
        "seed": seed,
        "window": window,
        "horizon": horizon,
        "best_epoch": best_epoch,
        "best_val_rul_rmse": best_val,
        "n_train": int(n_tr),
        "n_val": int(n_val),
        "n_test": int(len(rt)),
        "rul_rmse": float(mr["rmse"]),
        "rul_mae": float(mr["mae"]),
        "rul_r2": float(mr["r2"]),
        "soh_rmse": float(ms["rmse"]),
        "soh_mae": float(ms["mae"]),
        "soh_r2": float(ms["r2"]),
        **uq,
    }


def run_classical(data: dict, cell_id: str, info_level: str, corpus: str, window: int, horizon: int) -> list[dict]:
    x, y_rul = data["x"], data["y_rul"]
    n = int(len(x) * 0.8)
    x_te, y_te = x[n:], y_rul[n:]
    rows = []
    specs = [
        ("persist_fade", dict()),
        ("linear", dict()),
        ("persist_recursive", dict(n_train=n, x_full=x)),
        ("persist_hold", dict(y_rul_full=y_rul, n_train=n)),
        ("persist_mean", dict(y_rul_full=y_rul, n_train=n)),
    ]
    for method, extra in specs:
        m = evaluate_classical(x_te, y_te, method=method, **extra)
        m.update(
            {
                "experiment": "info_levels_v2_classical",
                "corpus": corpus,
                "battery": cell_id,
                "info_level": info_level,
                "seed": 0,
                "window": window,
                "horizon": horizon,
            }
        )
        rows.append(m)
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    metrics = (
        "rul_rmse",
        "rul_mae",
        "rul_r2",
        "soh_rmse",
        "soh_mae",
        "picp_rul",
        "pinaw_rul",
        "picp_soh",
        "pinaw_soh",
        "best_val_rul_rmse",
    )
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        row = {k: v for k, v in zip(keys, key)}
        row["n_runs"] = len(items)
        for m in metrics:
            vals = [float(it[m]) for it in items if m in it and it[m] is not None]
            if vals:
                row[m] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                row[f"{m}_all"] = vals
        out.append(row)
    return out


def stats_block(rows: list[dict]) -> list[dict]:
    """Paired L2 vs L0 (and L2 vs L1) tests on seed-level RMSE per cell/model."""
    by = defaultdict(list)
    for r in rows:
        by[(r["corpus"], r["model"], r["battery"], r["info_level"])].append(r)
    out = []
    for corpus, model, bat in sorted({(r["corpus"], r["model"], r["battery"]) for r in rows}):
        for a, b in (("L2", "L0"), ("L2", "L1"), ("L1", "L0")):
            ka = by.get((corpus, model, bat, a), [])
            kb = by.get((corpus, model, bat, b), [])
            if not ka or not kb:
                continue
            va = [float(x["rul_rmse"]) for x in sorted(ka, key=lambda z: z["seed"])]
            vb = [float(x["rul_rmse"]) for x in sorted(kb, key=lambda z: z["seed"])]
            n = min(len(va), len(vb))
            va, vb = va[:n], vb[:n]
            out.append(
                {
                    "corpus": corpus,
                    "model": model,
                    "battery": bat,
                    "contrast": f"{a}-vs-{b}",
                    "mean_a": float(np.mean(va)),
                    "mean_b": float(np.mean(vb)),
                    "delta_b_minus_a": float(np.mean(vb) - np.mean(va)),
                    "cohens_d": cohens_d(vb, va),
                    "wilcoxon_p": wilcoxon_p(va, vb),
                    "n_seeds": n,
                    "unit": "random seed (same cell), not cell-level n",
                    "values_a": va,
                    "values_b": vb,
                }
            )
    return out


def provenance_matrix() -> list[dict]:
    return [
        {"step": "raw NASA .mat / CALCE npy", "available_at_t": True, "target_future": False, "level": "source"},
        {"step": "discharge integration → Q_t", "available_at_t": True, "target_future": False, "level": "L0+"},
        {"step": "cap_norm = Q_t / Q_1 (this cell only)", "available_at_t": True, "target_future": False, "level": "L0+"},
        {"step": "mean_V / mean_I / mean_T / std_V (history ≤ t)", "available_at_t": True, "target_future": False, "level": "L0+"},
        {"step": "raw cycle index t", "available_at_t": True, "target_future": False, "level": "L0"},
        {"step": "L0 channel = t (unscaled)", "available_at_t": True, "target_future": False, "level": "L0"},
        {"step": "L0_clock channel = t/1000 (linear transform of t)", "available_at_t": True, "target_future": False, "level": "L0-sens"},
        {"step": "T_EOL := first cycle with SOH≤0.7", "available_at_t": False, "target_future": True, "level": "label+L2"},
        {"step": "L1 channel = t / median(T_EOL siblings)", "available_at_t": True, "target_future": False, "level": "L1"},
        {"step": "L2 channel = t / T_EOL^target", "available_at_t": False, "target_future": True, "level": "L2"},
        {"step": "RUL label = cycles until SOH≤0.7", "available_at_t": False, "target_future": True, "level": "label OK"},
        {"step": "sliding window start/step (same cell, stride 1)", "available_at_t": True, "target_future": False, "level": "L0+"},
        {"step": "no cross-cell windows", "available_at_t": True, "target_future": False, "level": "protocol"},
        {"step": "chronological 80/20; no joint all-cell scaler", "available_at_t": True, "target_future": False, "level": "protocol"},
        {"step": "early-stopping on train-time validation RUL RMSE", "available_at_t": True, "target_future": False, "level": "protocol"},
        {"step": "hyperparameters fixed a priori (not tuned on test)", "available_at_t": True, "target_future": False, "level": "protocol"},
        {"step": "no VMD/EMD on the test trajectory", "available_at_t": True, "target_future": False, "level": "protocol"},
    ]


def reporting_template_rows(agg: list[dict], classical: list[dict]) -> list[dict]:
    pers = {}
    for r in classical:
        if r["model"] == "persist_fade":
            pers[(r["corpus"], r["battery"])] = float(r["rul_rmse"])
    out = []
    for r in agg:
        out.append(
            {
                "model": r["model"],
                "corpus": r["corpus"],
                "cell": r["battery"],
                "info_level": r["info_level"],
                "cycle_channel": {
                    "L2": "t / T_EOL^target",
                    "L1": "t / median(T_EOL siblings)",
                    "L0": "raw t",
                    "L0_CLOCK": "t/1000",
                }.get(r["info_level"], r["info_level"]),
                "W": 50,
                "H": 5,
                "scaler": "per-cell features; no joint test-fit scaler",
                "rul_rmse": r.get("rul_rmse"),
                "rul_rmse_std": r.get("rul_rmse_std"),
                "soh_rmse": r.get("soh_rmse"),
                "persist_fade_rmse": pers.get((r["corpus"], r["battery"])),
            }
        )
    return out


def write_report(rows, classical, stats, diag, out_dir: Path) -> None:
    keys = ("experiment", "corpus", "model", "battery", "info_level")
    agg = aggregate(rows, keys)
    (out_dir / "agg.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    (out_dir / "classical.json").write_text(json.dumps(classical, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "reporting_template.json").write_text(
        json.dumps(reporting_template_rows(agg, classical), indent=2), encoding="utf-8"
    )
    lines = ["# Information-level v2 report\n\n", "T = EOL cycle (SOH ≤ 0.7). L0 = raw t.\n\n"]
    lines.append("## Lifetimes\n\n")
    for corpus, table in diag["lifetimes"].items():
        lines.append(f"### {corpus}\n\n| Cell | n | EOL70 | T_record | regen_sum |\n|---|---:|---:|---:|---:|\n")
        for k, v in table.items():
            lines.append(
                f"| {k} | {v['n_cycles']} | {v['eol70']} | {v['t_record']:.0f} | {v['regen_sum']:.3f} |\n"
            )
        lines.append("\n")
    lines.append("## Learned RUL RMSE (mean±std, seeds)\n\n")
    lines.append("| Corpus | Model | Cell | L2 | L1 | L0 |\n|---|---|---|---:|---:|---:|\n")
    pivot = defaultdict(dict)
    for r in agg:
        pivot[(r["corpus"], r["model"], r["battery"])][r["info_level"]] = (
            f"{r['rul_rmse']:.2f}±{r.get('rul_rmse_std', 0):.2f}"
        )
    for key in sorted(pivot):
        cells = pivot[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {cells.get('L2','-')} | "
            f"{cells.get('L1','-')} | {cells.get('L0','-')} |\n"
        )
    lines.append("\n## SOH RMSE\n\n| Corpus | Model | Cell | L2 | L1 | L0 |\n|---|---|---|---:|---:|---:|\n")
    sp = defaultdict(dict)
    for r in agg:
        if "soh_rmse" in r:
            sp[(r["corpus"], r["model"], r["battery"])][r["info_level"]] = (
                f"{r['soh_rmse']:.4f}±{r.get('soh_rmse_std', 0):.4f}"
            )
    for key in sorted(sp):
        cells = sp[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {cells.get('L2','-')} | "
            f"{cells.get('L1','-')} | {cells.get('L0','-')} |\n"
        )
    lines.append("\n## Classical RUL RMSE\n\n| Method | Corpus | Cell | RMSE |\n|---|---|---|---:|\n")
    seen = set()
    for r in classical:
        key = (r["model"], r["corpus"], r["battery"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {r['model']} | {r['corpus']} | {r['battery']} | {r['rul_rmse']:.2f} |\n")
    lines.append("\n## Seed-level tests (not cell-level n)\n\n")
    lines.append("| Contrast | Corpus | Model | Cell | ΔRMSE | Cohen d | Wilcoxon p | n |\n")
    lines.append("|---|---|---|---|---:|---:|---:|---:|\n")
    for s in stats:
        p = s["wilcoxon_p"]
        ps = f"{p:.4f}" if p is not None else "NA"
        lines.append(
            f"| {s['contrast']} | {s['corpus']} | {s['model']} | {s['battery']} | "
            f"{s['delta_b_minus_a']:.2f} | {s['cohens_d']:.2f} | {ps} | {s['n_seeds']} |\n"
        )
    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models", type=str, default="mlp,patch-tf")
    p.add_argument("--nasa-cells", type=str, default="B0005,B0006,B0007")
    p.add_argument("--calce-cells", type=str, default="CS2_35,CS2_36,CS2_37,CS2_38")
    p.add_argument("--severson-cells", type=str, default="")
    p.add_argument("--severson-n", type=int, default=24, help="stratified Severson cell count if --severson-cells empty")
    p.add_argument("--levels", type=str, default="L2,L1,L0")
    p.add_argument("--corpus", type=str, default="nasa,calce")
    p.add_argument(
        "--fit-protocol",
        type=str,
        default="within_cell",
        choices=["within_cell", "leave_target"],
        help="within_cell: Protocol A chronological 80/20 on target; "
        "leave_target: Protocol B train on siblings only, same target late-20%% test",
    )
    p.add_argument(
        "--loco-max-windows",
        type=int,
        default=20000,
        help="cap on concatenated sibling windows for leave_target",
    )
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 4)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    nasa_cells = [b.strip() for b in args.nasa_cells.split(",") if b.strip()]
    calce_cells = [b.strip() for b in args.calce_cells.split(",") if b.strip()]
    severson_cells = [b.strip() for b in args.severson_cells.split(",") if b.strip()]
    levels = [x.strip().upper() for x in args.levels.split(",") if x.strip()]
    corpora = [c.strip().lower() for c in args.corpus.split(",") if c.strip()]
    seeds = SEEDS if not args.smoke else [42]
    if args.smoke:
        nasa_cells = nasa_cells[:1]
        calce_cells = calce_cells[:1]
        models = models[:1]
        args.severson_n = min(args.severson_n, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "nasa"
    npy = ROOT / "data" / "calce" / "CALCE.npy"
    sev_dir = ROOT / "data" / "severson"
    sev_json = sev_dir / "qd_compact.json"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "info_levels_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    nasa_life = nasa_lifetime_table(data_dir) if "nasa" in corpora else {}
    calce_life = calce_lifetime_table(npy) if ("calce" in corpora and npy.exists()) else {}
    severson_life: dict[str, dict] = {}
    if "severson" in corpora:
        if not sev_json.exists():
            print("extracting Severson compact QD →", sev_json, flush=True)
            save_severson_compact(sev_dir, sev_json)
        severson_life = severson_lifetime_table(sev_json)
        if not severson_cells:
            severson_cells = stratify_severson_cells(severson_life, args.severson_n)
        if args.smoke:
            severson_cells = severson_cells[:3]
    sev_diag = {k: severson_life[k] for k in severson_cells if k in severson_life}
    diag = {"lifetimes": {"nasa": nasa_life, "calce": calce_life, "severson": sev_diag}}
    (out_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (out_dir / "provenance_matrix.json").write_text(
        json.dumps(provenance_matrix(), indent=2), encoding="utf-8"
    )

    jobs = []
    if "nasa" in corpora:
        for bat in nasa_cells:
            jobs.append(("nasa", bat, sibling_t_eol(nasa_life, bat)))
    if "calce" in corpora:
        for cid in calce_cells:
            if cid in calce_life:
                jobs.append(("calce", cid, sibling_t_eol(calce_life, cid)))
    if "severson" in corpora:
        # T_prior from ALL leave-target Severson cells (full D_train), not only the test subset.
        for cid in severson_cells:
            if cid in severson_life:
                jobs.append(("severson", cid, sibling_t_eol(severson_life, cid)))

    (out_dir / "sibling_scales.json").write_text(
        json.dumps({f"{c}:{b}": s for c, b, s in jobs}, indent=2), encoding="utf-8"
    )
    print(
        "device",
        device,
        "fit_protocol",
        args.fit_protocol,
        "jobs",
        [(c, b, s["t_target"], s["t_med"]) for c, b, s in jobs],
        flush=True,
    )

    rows: list[dict] = []
    classical: list[dict] = []
    raw_path = out_dir / "raw.json"
    (out_dir / "fit_protocol.txt").write_text(args.fit_protocol + "\n", encoding="utf-8")

    life_by_corpus = {"nasa": nasa_life, "calce": calce_life, "severson": severson_life}

    for corpus, cell_id, scales in jobs:
        for level in levels:
            ts = t_scale_for(level, scales)
            if corpus == "nasa":
                data = load_nasa_windows(data_dir, cell_id, args.window, args.horizon, level, ts)
            elif corpus == "calce":
                data = load_calce_rul_windows(npy, cell_id, args.window, args.horizon, level, ts)
            else:
                data = load_severson_rul_windows(
                    sev_json, cell_id, args.window, args.horizon, level, ts
                )
            data = restrict_pre_eol(data)
            train_pool = None
            if args.fit_protocol == "leave_target":
                life = life_by_corpus[corpus]
                # Full leave-target pool (119 for Severson), not only the scored subset.
                sib_ids = [k for k in life if k != cell_id]
                print(
                    f"  pooling leave-target siblings n={len(sib_ids)} for {corpus}:{cell_id}|{level}",
                    flush=True,
                )
                train_pool = pool_sibling_windows(
                    corpus,
                    cell_id,
                    sib_ids,
                    args.window,
                    args.horizon,
                    level,
                    life,
                    data_dir,
                    npy,
                    sev_json,
                    max_windows=args.loco_max_windows,
                    seed=42,
                )
                print(f"  pool windows={train_pool['n_pool_windows']}", flush=True)
            if level == levels[0]:
                classical.extend(
                    run_classical(data, cell_id, "level-invariant", corpus, args.window, args.horizon)
                )
            for model_name in models:
                for seed in seeds:
                    print(
                        f"=== {corpus} {model_name} {cell_id} {level} "
                        f"fit={args.fit_protocol} seed={seed} "
                        f"T_eol={scales['t_target']} T_med={scales['t_med']} t_scale={ts} ===",
                        flush=True,
                    )
                    res = train_one(
                        model_name,
                        cell_id,
                        level,
                        data,
                        args.epochs,
                        args.batch_size,
                        args.lr,
                        seed,
                        args.window,
                        args.horizon,
                        device,
                        corpus,
                        ts,
                        fit_protocol=args.fit_protocol,
                        train_pool=train_pool,
                    )
                    rows.append(res)
                    raw_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    stats = stats_block(rows)
    write_report(rows, classical, stats, diag, out_dir)
    print("done →", out_dir, flush=True)


if __name__ == "__main__":
    main()
