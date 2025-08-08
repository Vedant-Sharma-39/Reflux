import argparse
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Setup Project Root Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.io.data_loader import load_aggregated_data

def extract_env_name(env_def_json):
    """Extracts the 'name' from the env_definition JSON string."""
    try:
        env_def = json.loads(env_def_json)
        return env_def.get('name', 'unknown')
    except (json.JSONDecodeError, TypeError):
        return 'unknown'

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: Bet-Hedging in Asymmetric Environments."
    )
    parser.add_argument(
        "campaign_id", 
        default="fig5_asymmetric_patches",
        nargs='?',
        help="Campaign ID for the asymmetric patch experiment (default: fig5_asymmetric_patches)."
    )
    args = parser.parse_args()

    df = load_aggregated_data(args.campaign_id, project_root)
    if df is None or df.empty:
        sys.exit(f"Could not load data for campaign '{args.campaign_id}'. Aborting.")

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0
    df["env_name"] = df["env_definition"].apply(extract_env_name)

    # Define a clear order for the environments in the plot
    env_order = ["30_90", "60_60", "90_30", "scrambled_60_60"]
    df['env_name'] = pd.Categorical(df['env_name'], categories=env_order, ordered=True)

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=df,
        x="k_total",
        y="avg_front_speed",
        hue="phi",
        col="s",
        row="env_name",
        kind="line",
        marker="o",
        height=4,
        aspect=1.2,
        palette="coolwarm_r", # Reversed palette for this figure
        facet_kws={"margin_titles": True, "sharey": False},
        legend="full",
    )
    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Figure 5: Front Speed in Asymmetric Environments",
        y=1.03,
        fontsize=24,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure5_asymmetric_patches.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 5 saved to {output_path}")

if __name__ == "__main__":
    main()
