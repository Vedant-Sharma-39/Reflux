# FILE: scripts/paper_figures/fig5_asymmetric_patches.py (Corrected Data Combination)

import argparse
import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def extract_env_name(env_def_json):
    try:
        env_def = json.loads(env_def_json)
        return env_def.get("name", "unknown")
    except (json.JSONDecodeError, TypeError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure: Detailed Front Speed in Asymmetric Environments."
    )
    parser.add_argument("asymmetric_campaign")
    parser.add_argument("symmetric_campaign")
    args = parser.parse_args()
    project_root = get_project_root()

    path_asym = os.path.join(
        project_root,
        "data",
        args.asymmetric_campaign,
        "analysis",
        f"{args.asymmetric_campaign}_summary_aggregated.csv",
    )
    path_sym = os.path.join(
        project_root,
        "data",
        args.symmetric_campaign,
        "analysis",
        f"{args.symmetric_campaign}_summary_aggregated.csv",
    )
    output_path = os.path.join(
        project_root,
        "data",
        args.asymmetric_campaign,
        "analysis",
        "sup_fig_asymmetric_details.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df_asym = pd.read_csv(path_asym)
        df_sym = pd.read_csv(path_sym)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error: Missing or empty data file. {e}", file=sys.stderr)
        sys.exit(1)

    df_asym["s"] = df_asym["b_m"] - 1.0
    df_asym["env_name"] = df_asym["env_definition"].apply(extract_env_name)

    df_sym["s"] = df_sym["b_m"] - 1.0
    df_sym_60 = df_sym[df_sym["patch_width"] == 60].copy()
    df_sym_60["env_name"] = "60_60"

    plot_columns = ["s", "k_total", "phi", "avg_front_speed", "env_name"]
    df_plot = pd.concat(
        [df_asym[plot_columns], df_sym_60[plot_columns]], ignore_index=True
    )

    env_order = ["30_90", "60_60", "90_30", "scrambled_60_60"]
    df_plot["env_name"] = pd.Categorical(
        df_plot["env_name"], categories=env_order, ordered=True
    )

    sns.set_theme(style="ticks", context="talk")
    g = sns.relplot(
        data=df_plot,
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
    g.set_titles(row_template="{row_name}", col_template="s = {col_name:.2f}")
    g.fig.suptitle(
        "Supplementary Figure: Front Speed in Asymmetric Environments",
        y=1.03,
        fontsize=24,
    )
    g.legend.set_title(r"Bias, $\phi$")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1.02, 1))

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nAsymmetric Details Figure saved to {output_path}")


if __name__ == "__main__":
    main()
