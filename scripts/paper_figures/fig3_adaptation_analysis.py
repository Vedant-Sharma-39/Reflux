# FILE: scripts/paper_figures/fig3_adaptation_analysis.py

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


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
        description="Generate Figure 3: Adaptation to Environmental Asymmetry."
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
        "figure3_adaptation_plot.png",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df_asym = pd.read_csv(path_asym)
        df_sym = pd.read_csv(path_sym)
    except FileNotFoundError as e:
        print(f"Error: Missing a required data file. {e}", file=sys.stderr)
        sys.exit(1)

    df_asym["env_name"] = df_asym["env_definition"].apply(extract_env_name)
    df_sym_60 = df_sym[df_sym["patch_width"] == 60].copy()
    df_sym_60["env_name"] = "60_60"

    df = pd.concat([df_asym, df_sym_60], ignore_index=True)

    s_slice = -0.25
    df_slice = df[np.isclose(df["b_m"] - 1.0, s_slice)]

    opt_idx = df_slice.groupby("env_name")["avg_front_speed"].idxmax()
    df_opt = df_slice.loc[opt_idx].set_index("env_name")

    norm_speed = df_opt.loc["60_60"]["avg_front_speed"]
    df_opt["fitness_gain"] = df_opt["avg_front_speed"] / norm_speed
    df_opt = df_opt.reindex(["90_30", "60_60", "30_90", "scrambled_60_60"])

    sns.set_theme(style="ticks", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Optimal Strategy Adapts to Environmental Asymmetry (s = {s_slice})",
        fontsize=22,
        y=1.02,
    )

    axA = axes[0]
    sns.barplot(
        ax=axA, x=df_opt.index, y=df_opt["phi"], palette="coolwarm_r", edgecolor="black"
    )
    axA.axhline(0, color="black", ls="--", lw=1)
    axA.set_title("(a) Optimal Switching Bias", fontsize=16)
    axA.set_xlabel("Environment (WT Favored / M Favored Patch Width)")
    axA.set_ylabel(r"Optimal Bias, $\phi_{opt}$")

    axB = axes[1]
    sns.barplot(
        ax=axB,
        x=df_opt.index,
        y=df_opt["fitness_gain"],
        palette="crest",
        edgecolor="black",
    )
    axB.axhline(1.0, color="black", ls="--", lw=1)
    axB.set_title("(b) Maximal Fitness Gain", fontsize=16)
    axB.set_xlabel("Environment (WT Favored / M Favored Patch Width)")
    axB.set_ylabel("Max Fitness / Symmetric Fitness")

    sns.despine(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"\nAdaptation Figure 3 saved to {output_path}")


if __name__ == "__main__":
    main()
