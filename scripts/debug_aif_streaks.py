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
import matplotlib # Added for publication settings

# --- Publication Settings ---
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial"] # Use common sans-serif fonts

# ---- Display knobs ----
SMOOTHING_MODE = "ema"   # {"off","ema","ma"}
EMA_ALPHA      = 0.25
MA_WINDOW      = 7
DROP_LAST_N    = 2       # drop last N points per curve (avoid end-step wiggles)
# --- CHANGE: Update output filename and format ---
SAVE_FIG_NAME  = "fig_aif_sector_trajectories_and_colony.pdf" # Changed to PDF for vector graphics
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.model_aif import AifModelSimulation, Resistant, Susceptible # Added Susceptible
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
    print("--- Generating AIF Sector Trajectory Figure ---") # Changed title

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
        "b_res": 0.97, # Set specific parameter for plot titles
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

    output_dir = PROJECT_ROOT / "figures" / "debug_runs" # Keep debug output separate
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

    if "root_sid" not in df_all.columns:
        df_all["root_sid"] = df_all["sid"]

    max_r = float(df_all["radius"].max())
    df_last = df_all[np.abs(df_all["radius"] - max_r) < 1e-9]
    survivor_roots = set(df_last.loc[df_last["type"] == Resistant, "root_sid"].astype(int).tolist())

    df_series = df_all[
        (df_all["type"] == Resistant) &
        (df_all["root_sid"].isin(survivor_roots))
    ][["root_sid","radius","width_cells"]].rename(columns={"root_sid":"sid"})

    def drop_tail(g):
        if DROP_LAST_N > 0 and len(g) > DROP_LAST_N:
            return g.iloc[:-DROP_LAST_N, :]
        return g

    df_pop["is_front"] = compute_front_mask(df_pop)
    final_red_front = int(((df_pop["is_front"] == True) & (df_pop["type"] == Resistant)).sum())
    online_final_sum = int(df_last[df_last["type"] == Resistant]["width_cells"].sum())
    print(f"[Sanity Check] Final Online Sum (Resistant Widths): {online_final_sum} | "
          f"Final Lattice Resistant Front Cells: {final_red_front}")

    # 3) plot
    print("\n[3/3] Plotting for publication...")
    # --- CHANGE: Set theme for publication ---
    sns.set_theme(style="ticks", context="paper") # Use 'paper' context for smaller fonts/elements
    fig, (axA, axB) = plt.subplots(
        1, 2,
        figsize=(10, 5), # Adjusted size for a standard paper width
        gridspec_kw={"width_ratios":[0.8, 1.0]}, # Give slightly more space to panel B
        constrained_layout=True # Use constrained layout for better spacing
    )
    # --- REMOVED: fig.suptitle removed for typical publication style ---

    # --- Panel A: Full colony visualization ---
    # Define colors explicitly for publication
    COLOR_SUSCEPTIBLE = "#bfc5ca" # Light grey
    COLOR_RESISTANT = "#e63946"   # Red

    df_res = df_pop[df_pop["type"] == Resistant]
    df_sus = df_pop[df_pop["type"] == Susceptible] # Assuming type 1 is Susceptible
    axA.scatter(df_sus["x"], df_sus["y"], s=0.5, c=COLOR_SUSCEPTIBLE, alpha=0.6, linewidths=0, rasterized=True) # Rasterize for smaller file size in PDF
    axA.scatter(df_res["x"],  df_res["y"],  s=1.5, c=COLOR_RESISTANT, alpha=0.9, linewidths=0, rasterized=True)
    axA.set_aspect("equal", "box")
    # --- CHANGE: Improved Panel A title and labels ---
    axA.set_title("(A) Final Colony State", fontsize=12, weight='bold')
    axA.set_xlabel("Spatial Coordinate X", fontsize=10)
    axA.set_ylabel("Spatial Coordinate Y", fontsize=10)
    axA.tick_params(axis='both', which='major', labelsize=8) # Smaller tick labels
    axA.tick_params(left=False, right=False , labelleft=False ,
                    labelbottom=False, bottom=False) # Hide ticks and labels for cleaner look

    # --- Panel B: Survivor Trajectories ---
    num_survivors = df_series['sid'].nunique()
    # --- CHANGE: Use a perceptually uniform colormap ---
    palette = sns.color_palette("viridis", n_colors=num_survivors)

    # Plot raw data faintly first
    for i, (sid, g) in enumerate(df_series.groupby("sid")):
        g = g.sort_values("radius")
        g = drop_tail(g)
        axB.plot(g["radius"], g["width_cells"], linewidth=1.0, alpha=0.25, color=palette[i])

    # Plot smoothed data more prominently
    plotted_sids = [] # Keep track to avoid duplicate labels
    for i, (sid, g) in enumerate(df_series.groupby("sid")):
        g = g.sort_values("radius")
        g = drop_tail(g)
        if SMOOTHING_MODE != "off":
            g_s = smooth_series(g, "radius", "width_cells", SMOOTHING_MODE)
            # Only add label once per sid
            label = f"Lineage {sid}" if sid not in plotted_sids else None
            axB.plot(g_s["radius"], g_s["width_cells"], linewidth=1.8, color=palette[i], label=label)
            if label: plotted_sids.append(sid)

    # --- CHANGE: Improved Panel B title and labels ---
    axB.set_xlabel("Colony Radius ($r$)", fontsize=10)
    axB.set_ylabel("Sector Width ($W$, cell count)", fontsize=10)
    smoothing_label = ""
    if SMOOTHING_MODE == "ema":
        smoothing_label = f" (EMA smoothed, $\\alpha={EMA_ALPHA}$)"
    elif SMOOTHING_MODE == "ma":
         smoothing_label = f" (MA smoothed, window={MA_WINDOW})"
    axB.set_title(f"(B) Resistant Sector Trajectories{smoothing_label}", fontsize=12, weight='bold')

    axB.grid(True, which="major", linestyle=":", linewidth=0.5, color='grey') # Make grid lighter
    axB.tick_params(axis='both', which='major', labelsize=8) # Smaller tick labels
    axB.set_xlim(left=params["initial_droplet_radius"]*0.95) # Start x-axis near initial radius
    axB.set_ylim(bottom=0) # Start y-axis at 0

    # --- CHANGE: Improved legend ---
    if plotted_sids: # Only show legend if smoothed lines were plotted
        handles, labels = axB.get_legend_handles_labels()
        # Keep legend small and potentially outside if many lines
        if num_survivors <= 10:
             axB.legend(handles=handles, labels=labels, loc='upper left', fontsize=8, title="Survivor Lineage ID", title_fontsize=9)
        else:
             # For many lines, place legend outside
             axB.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title="Survivor Lineage ID", title_fontsize=9)

    sns.despine(fig=fig) # Remove top and right spines

    # --- CHANGE: Save to final figures directory, not debug ---
    final_figure_dir = PROJECT_ROOT / "figures"
    final_figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = final_figure_dir / SAVE_FIG_NAME

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ figure saved to: {out_path}")
    # plt.show() # Keep commented out for automated runs

if __name__ == "__main__":
    main()