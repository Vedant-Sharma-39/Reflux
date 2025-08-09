# FILE: scripts/paper_figures/fig5_asymmetric_patches.py (Standardized & Robust Version)
import argparse
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def extract_env_name(env_def_json):
    try:
        # The parameter from the CSV might be a string representation of a dict
        if isinstance(env_def_json, str):
            env_def = json.loads(env_def_json)
        else:  # Or it might already be a dict
            env_def = env_def_json
        return env_def.get("name", "unknown")
    except (json.JSONDecodeError, TypeError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: Bet-Hedging in Asymmetric Environments."
    )
    parser.add_argument("campaign_id")
    args = parser.parse_args()
    project_root = get_project_root()
    summary_path = os.path.join(
        project_root,
        "data",
        args.campaign_id,
        "analysis",
        f"{args.campaign_id}_summary_aggregated.csv",
    )

    try:
        df = pd.read_csv(summary_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "figure5_asymmetric_patches.png")

    if df.empty:
        print(
            f"Warning: No data found for campaign '{args.campaign_id}'. Cannot generate Figure 5."
        )
        fig, _ = plt.subplots()
        fig.text(0.5, 0.5, "Figure 5: No Data Available", ha="center", va="center")
        plt.savefig(output_path, dpi=300)
        sys.exit(0)

    print(f"Loaded {len(df)} simulation results.")
    df["s"] = df["b_m"] - 1.0
    df["env_name"] = df["env_definition"].apply(extract_env_name)
    env_order = ["30_90", "60_60", "90_30", "scrambled_60_60"]
    df["env_name"] = pd.Categorical(df["env_name"], categories=env_order, ordered=True)

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
        palette="coolwarm_r",
        facet_kws={"margin_titles": True, "sharey": False},
        legend="full",
    )
    g.set_xlabels(r"Switching Rate, $k_{total}$")
    g.set_ylabels("Mean Front Speed, $v$")
    g.set(xscale="log")
    g.fig.suptitle(
        "Figure 5: Front Speed in Asymmetric Environments", y=1.03, fontsize=24
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure 5 saved to {output_path}")


if __name__ == "__main__":
    main()
