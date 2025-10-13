# FILE: scripts/debug_aif_streaks.py
import sys
from pathlib import Path
import json
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---- Display knobs ----
SMOOTHING_MODE = "ema"   # {"off","ema","ma"}
EMA_ALPHA      = 0.25
MA_WINDOW      = 7
DROP_LAST_N    = 2       # drop last N points per curve (avoid end-step wiggles)
SAVE_FIG_NAME  = "debug_aif_online_only_with_full_colony.png"
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_aif import AifModelSimulation, Resistant
from src.utils.analysis_helpers import load_population_data

CUBE_DIRS = [(1,-1,0),(1,0,-1),(0,1,-1),(-1,1,0),(-1,0,1),(0,-1,1)]

def axial_to_cartesian(q: int, r: int, size: float = 1.0):
    x = size * (1.5 * q)
    y = size * ((np.sqrt(3)/2.0) * q + np.sqrt(3) * r)
    return x, y

def radius_xy(q: int, r: int) -> float:
    x, y = axial_to_cartesian(q, r)
    return float(np.hypot(x, y))

def neighbors_axial(q: int, r: int):
    for dq, dr, ds in CUBE_DIRS:
        yield (q + dq, r + dr)

def compute_front_mask(df_all: pd.DataFrame) -> pd.Series:
    """True front: has ≥1 outward (larger-radius) empty neighbor."""
    occ = set(zip(df_all.q.values.tolist(), df_all.r.values.tolist()))
    def is_front_row(row) -> bool:
        q, r = int(row.q), int(row.r)
        r0 = radius_xy(q, r)
        for qn, rn in neighbors_axial(q, r):
            if (qn, rn) not in occ and radius_xy(qn, rn) > r0:
                return True
        return False
    return df_all.apply(is_front_row, axis=1)

def smooth_series(df: pd.DataFrame, xcol: str, ycol: str, mode: str) -> pd.DataFrame:
    g = df.sort_values(xcol).copy()
    if mode == "ema":
        y = g[ycol].to_numpy(dtype=float)
        if len(y) > 0:
            out = np.empty_like(y, dtype=float)
            out[0] = y[0]
            a = float(EMA_ALPHA)
            for i in range(1, len(y)):
                out[i] = (1.0 - a) * out[i-1] + a * y[i]
            g[ycol] = out
        return g
    elif mode == "ma":
        if MA_WINDOW <= 1 or len(g) < 2: return g
        g[ycol] = g[ycol].rolling(window=int(MA_WINDOW), min_periods=1, center=True).mean()
        return g
    else:
        return g

def main():
    print("--- AIF Sector Width: ONLINE only + full-colony view (red cells highlighted) ---")

    # 1) simulate
    print("\n[1/3] Running simulation...")
    initial_radius = 300
    params = {
        "campaign_id": "debug_aif_streaks",
        "simulation_class": "AifModelSimulation",

        "initial_condition_type": "aif_front_bands",
        "band_width": 12,
        "num_bands": 18,

        "initial_droplet_radius": initial_radius,
        "max_steps": 250_000,

        "b_sus": 1.0,
        "b_res": 0.97,
        "b_comp": 1.0,
        "k_res_comp": 0.0,

        # ΔR logging
        "sector_metrics_dr": 1.0,
        "sector_metrics_interval": 0,

        # engine-level denoising & persistence (tunable)
        "front_denoise_window": 5,
        "min_island_len": 3,
        "sid_iou_thresh": 0.15,
        "sid_center_delta": 0.10,
        "death_hysteresis": 3,
    }

    output_dir = PROJECT_ROOT / "figures" / "debug_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_pop_file = output_dir / "aif_streaks_for_debug.json.gz"

    sim = AifModelSimulation(**params)
    with tqdm(total=params["max_steps"], desc="Simulating") as pbar:
        while sim.step_count < params["max_steps"]:
            active, _ = sim.step()
            pbar.update(1)
            if not active:
                print("\nSimulation ended (no active events).")
                break

    print(f"\nSimulation finished at step {sim.step_count}.")
    final_pop_data = [{"q": int(h.q), "r": int(h.r), "type": int(t)} for h, t in sim.population.items()]
    with gzip.open(temp_pop_file, "wt", encoding="utf-8") as f:
        json.dump(final_pop_data, f)
    print(f"Final population data saved to: {temp_pop_file}")

    # 2) load + build series using **root_sid** (no stitching needed)
    print("\n[2/3] Preparing survivor series (root_sid based)...")
    df_pop = load_population_data(temp_pop_file)
    if "x" not in df_pop.columns or "y" not in df_pop.columns:
        xs, ys = [], []
        for q, r in df_pop[["q","r"]].itertuples(index=False):
            x, y = axial_to_cartesian(int(q), int(r))
            xs.append(x); ys.append(y)
        df_pop["x"], df_pop["y"] = xs, ys

    df_all = pd.DataFrame(sim.sector_traj_log)
    if df_all.empty:
        print("Warning: no in-run sector logs found.")
        return

    # backward-compat guard (if an older model without root_sid is used)
    if "root_sid" not in df_all.columns:
        df_all["root_sid"] = df_all["sid"]

    # survivors by **max radius** snapshot, using root_sid
    max_r = float(df_all["radius"].max())
    df_last = df_all[np.abs(df_all["radius"] - max_r) < 1e-9]
    survivor_roots = set(df_last.loc[df_last["type"] == Resistant, "root_sid"].astype(int).tolist())

    # keep only Resistant rows from survivor root lineages
    df_series = df_all[
        (df_all["type"] == Resistant) &
        (df_all["root_sid"].isin(survivor_roots))
    ][["root_sid","radius","width_cells"]].rename(columns={"root_sid":"sid"})

    # optional last-N drop to avoid tail wiggles
    def drop_tail(g):
        if DROP_LAST_N > 0 and len(g) > DROP_LAST_N:
            return g.iloc[:-DROP_LAST_N, :]
        return g

    # parity sanity
    df_pop["is_front"] = compute_front_mask(df_pop)
    final_red_front = int(((df_pop["is_front"] == True) & (df_pop["type"] == Resistant)).sum())
    online_final_sum = int(df_last[df_last["type"] == Resistant]["width_cells"].sum())
    print(f"[sanity] final online sum (red widths): {online_final_sum} | "
          f"final lattice red front cells: {final_red_front}")

    # 3) plot
    print("\n[3/3] Plotting...")
    sns.set_theme(style="whitegrid", context="talk")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios":[1.05, 1.25]})
    fig.suptitle("Sector width vs radius — survivors only (online, root_sid) + full-colony view of red cells", y=0.98)

    # A: full colony
    df_red  = df_pop[df_pop["type"] == Resistant]
    df_grey = df_pop[df_pop["type"] != Resistant]
    axA.scatter(df_grey["x"], df_grey["y"], s=1, c="#bfc5ca", alpha=0.6, linewidths=0)
    axA.scatter(df_red["x"],  df_red["y"],  s=3, c="#e63946", alpha=0.9, linewidths=0)
    axA.set_aspect("equal", "box")
    axA.set_title("Full colony (final lattice): red cells highlighted")
    axA.set_xlabel("x"); axA.set_ylabel("y")

    # B: survivor series grouped by root_sid
    for sid, g in df_series.groupby("sid"):
        g = g.sort_values("radius")
        g = drop_tail(g)
        (ln,) = axB.plot(g["radius"], g["width_cells"], linewidth=1.5, alpha=0.35, label=f"sid {sid} (raw)")
        color = ln.get_color()
        if SMOOTHING_MODE != "off":
            g_s = smooth_series(g, "radius", "width_cells", SMOOTHING_MODE)
            axB.plot(g_s["radius"], g_s["width_cells"], linewidth=2.4, color=color, label=f"sid {sid} ({SMOOTHING_MODE})")

    axB.set_xlabel("Radius")
    axB.set_ylabel("Sector Width on Front (cells)")
    ttl = "" if SMOOTHING_MODE == "off" else f" (smoothed: {SMOOTHING_MODE})"
    axB.set_title(f"Survivor red sectors — Online widths{ttl}")
    axB.grid(True, which="both", linestyle=":")
    axB.legend(ncol=2, fontsize=10)

    out_img = output_dir / SAVE_FIG_NAME
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"\n✅ Figure saved to: {out_img}")
    plt.show()


if __name__ == "__main__":
    main()
