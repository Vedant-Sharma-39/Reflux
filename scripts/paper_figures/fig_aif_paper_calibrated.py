# FILE: scripts/paper_figures/fig_aif_paper_calibrated.py
#!/usr/bin/env python3
"""
One-click paper-style AIF figures (robust param merge).

It will:
  • Auto-resolve the calibrated campaigns from src.config (singlesector & rimbands).
  • Load sector trajectories and final populations.
  • Merge trajectories with per-task parameters by joining on task_id/run_id
    using either the consolidated summary CSV or the master task list JSONL.
  • Produce:
      - colony overlay
      - survivor curves (faceted by width/band_width)
      - median & IQR (faceted by width/band_width)

Run:
  python scripts/paper_figures/fig_aif_paper_calibrated.py
"""

from __future__ import annotations
import gzip, json, math, re, sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Project + config discovery
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.config import EXPERIMENTS  # will raise if missing

def _resolve_campaign(exp_key_hint: str, campaign_hint: str) -> Optional[str]:
    if exp_key_hint in EXPERIMENTS:
        return EXPERIMENTS[exp_key_hint]["campaign_id"]
    for k, v in EXPERIMENTS.items():
        if v.get("campaign_id") == campaign_hint:
            return v["campaign_id"]
    for k, v in EXPERIMENTS.items():
        if exp_key_hint in k:
            return v["campaign_id"]
    return None

CAMPAIGNS = {
    "singlesector": _resolve_campaign("aif_paper_singlesector_calibrated", "aif_singlesector_cal_v1"),
    "rimbands":     _resolve_campaign("aif_paper_rimbands_calibrated",     "aif_rimbands_cal_v1"),
}

CELL_SIZE_UM = 5.0  # μm / cell

# ---------------------------
# IO helpers
# ---------------------------
def campaign_paths(campaign: str) -> Dict[str, Path]:
    base = PROJECT_ROOT / "data" / campaign
    d = {
        "base": base,
        "raw": base / "raw",
        "traj": base / "trajectories",
        "traj_raw": base / "trajectories_raw",
        "pop": base / "populations",
        "analysis": base / "analysis",
        "fig": PROJECT_ROOT / "figures" / campaign,
    }
    d["fig"].mkdir(parents=True, exist_ok=True)
    return d

def _try_load_gz_json(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)

NEEDED_COLS = {"step","radius","sid","root_sid","type","width_cells","start_theta","end_theta"}

def load_sector_traj(paths: Dict[str, Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    # worker-saved gz trajectories
    for gdir in [paths["traj"], paths["traj_raw"]]:
        if not gdir.exists(): continue
        for p in sorted(list(gdir.glob("traj_*.json.gz")) + list(gdir.glob("traj_sector_*.json.gz"))):
            try:
                obj = _try_load_gz_json(p)
                rows = obj if isinstance(obj, list) else obj.get("sector_trajectory", obj.get("sector_traj_log", []))
                if not rows: continue
                df = pd.DataFrame(rows)
                if not NEEDED_COLS.issubset(df.columns): continue
                # run_id = task_id embedded in filename
                rid = p.stem.replace("traj_","").replace("traj_sector_","")
                df["run_id"] = rid
                frames.append(df)
            except Exception:
                continue
    # raw JSONL fallback (pre-consolidation)
    if not frames and paths["raw"].exists():
        for jl in sorted(paths["raw"].glob("chunk_*.jsonl")):
            with jl.open("r") as f:
                for line in f:
                    try:
                        dat = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rows = dat.get("sector_trajectory") or dat.get("sector_traj_log")
                    if not rows: continue
                    df = pd.DataFrame(rows)
                    if not NEEDED_COLS.issubset(df.columns): continue
                    df["run_id"] = str(dat.get("task_id", jl.stem))
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_population(paths: Dict[str, Path]) -> Optional[pd.DataFrame]:
    pop_dir = paths["pop"]
    if not pop_dir.exists(): return None
    pops = sorted(pop_dir.glob("pop_*.json.gz"))
    if not pops: return None
    try:
        obj = _try_load_gz_json(pops[0])
        return pd.DataFrame(obj)
    except Exception:
        return None

# ---------------------------
# Param sources (robust)
# ---------------------------
def load_param_map(paths: Dict[str, Path], campaign: str) -> pd.DataFrame:
    """
    Return a DataFrame with one row per task_id/run_id and columns:
    ['run_id','b_res','sector_width_initial','band_width','replicate', ...]
    Priority: consolidated summary CSV -> master_tasks.jsonl
    """
    # 1) consolidated CSV
    summary_csv = paths["analysis"] / f"{campaign}_summary_aggregated.csv"
    cols_we_want = ["task_id","b_res","sector_width_initial","band_width","num_bands","replicate"]
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv, low_memory=False)
            have = [c for c in cols_we_want if c in df.columns]
            if "task_id" in df.columns and have:
                out = df[["task_id"] + have].drop_duplicates("task_id")
                out = out.rename(columns={"task_id":"run_id"})
                out["run_id"] = out["run_id"].astype(str)
                return out
        except Exception:
            pass
    # 2) master task list
    mfile = paths["base"] / f"{campaign}_master_tasks.jsonl"
    rows = []
    if mfile.exists():
        with mfile.open("r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(d.get("task_id"))
                if not rid: continue
                row = {"run_id": rid}
                for k in ("b_res","sector_width_initial","band_width","num_bands","replicate"):
                    if k in d: row[k] = d[k]
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["run_id"])

def attach_params_by_merge(df_traj: pd.DataFrame, paths: Dict[str, Path], campaign: str) -> pd.DataFrame:
    """
    Merge params to trajectories using run_id (task_id). If some columns are still missing,
    try very light regex parsing from run_id as a last resort.
    """
    if df_traj.empty: return df_traj
    pm = load_param_map(paths, campaign)
    df = df_traj.merge(pm, on="run_id", how="left")

    # Last-resort tiny parser (only if still missing)
    need_any = (("sector_width_initial" not in df.columns) or df["sector_width_initial"].isna().all()) and \
               (("band_width" not in df.columns) or df["band_width"].isna().all())
    if need_any:
        NUM = r"(-?\d+(?:p\d+)?)"
        rxes = [
            ("b_res", re.compile(rf"\bb_res{NUM}\b")),
            ("sector_width_initial", re.compile(rf"\bsector_width_initial(\d+)\b")),
            ("band_width", re.compile(rf"\bband_width(\d+)\b")),
        ]
        meta = []
        for rid in df["run_id"].unique():
            md = {"run_id": rid}
            for k, rx in rxes:
                m = rx.search(rid)
                if not m: continue
                val = m.group(1)
                if k == "b_res": md[k] = float(val.replace("p","."))
                else: md[k] = int(val)
            meta.append(md)
        if meta:
            mdf = pd.DataFrame(meta)
            df = df.drop(columns=[c for c in ("b_res","sector_width_initial","band_width") if c in df.columns])
            df = df.merge(mdf, on="run_id", how="left")
    return df

# ---------------------------
# Geometry & filters
# ---------------------------
SUS, RES = 1, 2
def axial_to_xy(q: int, r: int):
    x = 1.5 * q
    y = (np.sqrt(3)/2.0) * q + np.sqrt(3) * r
    return x, y
def compute_xy_for_pop(df_pop: pd.DataFrame) -> pd.DataFrame:
    if "x" in df_pop.columns and "y" in df_pop.columns: return df_pop
    xs, ys = [], []
    for q, r in df_pop[["q","r"]].itertuples(index=False):
        x, y = axial_to_xy(int(q), int(r))
        xs.append(x); ys.append(y)
    out = df_pop.copy(); out["x"], out["y"] = xs, ys
    return out
def survivors_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    max_r = float(df["radius"].max())
    at_end = df[np.abs(df["radius"] - max_r) < 1e-9]
    roots = set(at_end.loc[at_end["type"] == RES, "root_sid"].astype(int).tolist())
    return df[(df["type"] == RES) & (df["root_sid"].isin(roots))]
def ema(y: np.ndarray, a: float = 0.25) -> np.ndarray:
    if len(y) == 0: return y
    out = np.empty_like(y, dtype=float); out[0] = y[0]
    for i in range(1, len(y)): out[i] = (1.0 - a) * out[i-1] + a * y[i]
    return out

# ---------------------------
# Plotters
# ---------------------------
def colony_overlay(df_pop: pd.DataFrame, out_png: Path):
    df_pop = compute_xy_for_pop(df_pop)
    fig, ax = plt.subplots(figsize=(8, 8))
    df_r = df_pop[df_pop["type"] == RES]; df_g = df_pop[df_pop["type"] != RES]
    ax.scatter(df_g["x"], df_g["y"], s=1, c="#bfc5ca", alpha=0.6, linewidths=0)
    ax.scatter(df_r["x"], df_r["y"], s=3, c="#e63946", alpha=0.9, linewidths=0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    ax.set_title("Final colony (red = resistant)")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)

def survivor_curves_panel(df: pd.DataFrame, key: str, out_png: Path, title_prefix: str):
    if df.empty: return
    if key not in df.columns:
        print(f"[warn] '{key}' column missing; skipping survivor curves panel."); return
    levels = [x for x in sorted(df[key].dropna().unique())]
    if not levels: return
    n = len(levels); cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows), squeeze=False)
    for ax, lev in zip(axes.flat, levels):
        sub = df[df[key] == lev]
        if sub.empty: ax.set_visible(False); continue
        for sid, g in sub.groupby("root_sid"):
            g = g.sort_values("radius")
            if len(g) > 2: g = g.iloc[:-2, :]
            x = g["radius"].to_numpy(); y = g["width_cells"].to_numpy()
            (ln,) = ax.plot(x, y, alpha=0.25, linewidth=1.0)
            ax.plot(x, ema(y, 0.25), color=ln.get_color(), linewidth=1.8)
        ax.set_xlabel("radius (cells)"); ax.set_ylabel("width (cells)")
        ax.set_title(f"{key}={lev}")
        ax.grid(True, linestyle=":", alpha=0.5)
        sec = ax.secondary_xaxis("top", functions=(lambda c: c*CELL_SIZE_UM, lambda u: u/CELL_SIZE_UM))
        sec.set_xlabel("radius (μm)")
    for j in range(len(levels), rows * cols):
        fig.delaxes(axes.flat[j])
    fig.suptitle(f"{title_prefix} — Survivor curves (facet by {key})", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300); plt.close(fig)

def median_iqr_panel(df: pd.DataFrame, key: str, out_png: Path, bin_dr: float, title_prefix: str):
    if df.empty: return
    if key not in df.columns:
        print(f"[warn] '{key}' column missing; skipping median/IQR panel."); return
    levels = [x for x in sorted(df[key].dropna().unique())]
    if not levels: return
    n = len(levels); cols = min(3, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows), squeeze=False)
    for ax, lev in zip(axes.flat, levels):
        sub = df[df[key] == lev]
        if sub.empty: ax.set_visible(False); continue
        rmin, rmax = float(sub["radius"].min()), float(sub["radius"].max())
        bins = np.arange(math.floor(rmin), math.ceil(rmax) + bin_dr, bin_dr)
        sub = sub.copy()
        sub["rbin"] = pd.cut(sub["radius"], bins=bins, include_lowest=True, right=False)
        sub["rbin_c"] = sub["rbin"].apply(lambda x: 0.5*(x.left+x.right) if pd.notnull(x) else np.nan)
        per_run = (sub.groupby(["run_id","rbin_c"])["width_cells"].median().reset_index().dropna(subset=["rbin_c"]))
        if per_run.empty:
            ax.set_visible(False); continue
        agg = per_run.groupby("rbin_c")["width_cells"].agg(
            med="median",
            q25=lambda s: np.percentile(s, 25),
            q75=lambda s: np.percentile(s, 75),
        ).reset_index()
        ax.plot(agg["rbin_c"], agg["med"], linewidth=2.0)
        ax.fill_between(agg["rbin_c"], agg["q25"], agg["q75"], alpha=0.25)
        ax.set_xlabel("radius (cells)"); ax.set_ylabel("width (cells)")
        ax.set_title(f"{key}={lev}")
        ax.grid(True, linestyle=":", alpha=0.5)
        sec = ax.secondary_xaxis("top", functions=(lambda c: c*CELL_SIZE_UM, lambda u: u/CELL_SIZE_UM))
        sec.set_xlabel("radius (μm)")
    for j in range(len(levels), rows * cols):
        fig.delaxes(axes.flat[j])
    fig.suptitle(f"{title_prefix} — Median & IQR (facet by {key})", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300); plt.close(fig)

def simple_curves(df: pd.DataFrame, title_bits: str, out_png: Path):
    if df.empty: return
    fig, ax = plt.subplots(figsize=(9, 6))
    for sid, g in df.groupby("root_sid"):
        g = g.sort_values("radius")
        if len(g) > 2: g = g.iloc[:-2, :]
        x = g["radius"].to_numpy(); y = g["width_cells"].to_numpy()
        (ln,) = ax.plot(x, y, alpha=0.25, linewidth=1.0)
        ax.plot(x, ema(y, 0.25), color=ln.get_color(), linewidth=1.8)
    ax.set_xlabel("radius (cells) — top: μm")
    ax.set_ylabel("width (cells)")
    ax.set_title(f"Survivor curves {title_bits}")
    ax.grid(True, linestyle=":", alpha=0.5)
    sec = ax.secondary_xaxis("top", functions=(lambda c: c*CELL_SIZE_UM, lambda u: u/CELL_SIZE_UM))
    sec.set_xlabel("radius (μm)")
    fig.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)

# ---------------------------
# Runner
# ---------------------------
def run_campaign(kind: str, campaign: str):
    print(f"\n=== {kind.upper()} :: {campaign} ===")
    paths = campaign_paths(campaign)

    df_traj = load_sector_traj(paths)
    if df_traj.empty:
        print(f"[warn] No sector trajectories for '{campaign}'. If jobs just finished, run:")
        print(f"       python manage.py consolidate {campaign}")
        return

    # Merge params by run_id <- task_id
    df_traj = attach_params_by_merge(df_traj, paths, campaign)
    if "b_res" not in df_traj.columns:
        print("[warn] b_res not found in params; proceeding without it.")

    df_traj = survivors_only(df_traj)

    # Colony overlay
    df_pop = load_population(paths)
    if df_pop is not None and not df_pop.empty:
        out = paths["fig"] / f"{campaign}_colony_overlay.png"
        colony_overlay(df_pop, out); print("  saved:", out)

    # Choose facet key by mode
    key = "sector_width_initial" if kind == "singlesector" else "band_width"

    survivor_curves_panel(df_traj, key, paths["fig"] / f"{campaign}_survivor_curves_facet_{key}.png",
                          title_prefix=("Single sector" if kind=="singlesector" else "Rim bands"))
    print("  saved:", paths["fig"] / f"{campaign}_survivor_curves_facet_{key}.png")

    median_iqr_panel(df_traj, key, paths["fig"] / f"{campaign}_median_iqr_facet_{key}.png",
                     bin_dr=5.0, title_prefix=("Single sector" if kind=="singlesector" else "Rim bands"))
    print("  saved:", paths["fig"] / f"{campaign}_median_iqr_facet_{key}.png")

    # Optional single-condition plot at dominant b_res (if available)
    if "b_res" in df_traj.columns and not df_traj["b_res"].dropna().empty:
        dom_b = df_traj["b_res"].value_counts().idxmax()
        sub = df_traj[np.isclose(df_traj["b_res"].astype(float), float(dom_b), atol=1e-12)].copy()
        title_bits = f"(b_res={dom_b})"
        out = paths["fig"] / f"{campaign}_survivor_curves_bres_{str(dom_b).replace('.','p')}.png"
        simple_curves(sub, title_bits, out)
        print("  saved:", out)

def main():
    any_ok = False
    for kind in ("singlesector", "rimbands"):
        camp = CAMPAIGNS.get(kind)
        if not camp:
            print(f"[skip] Could not resolve campaign for {kind}. Add the calibrated experiments to src.config.")
            continue
        run_campaign(kind, camp); any_ok = True
    if not any_ok:
        raise SystemExit("[fatal] No calibrated campaigns resolved. Please add them to src.config and rerun.")

if __name__ == "__main__":
    main()
