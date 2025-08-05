import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from tqdm import tqdm

def load_timeseries_data(ts_db_path, task_ids_to_load):
    """Loads specific timeseries from the gzipped JSONL database."""
    ts_data = {task_id: [] for task_id in task_ids_to_load}
    with gzip.open(ts_db_path, 'rt') as f:
        for line in tqdm(f, desc="Loading timeseries DB"):
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                if task_id in task_ids_to_load:
                    ts_data[task_id].append(pd.DataFrame(data.get('timeseries', [])))
            except json.JSONDecodeError:
                continue
    # Concatenate data for each task_id
    for task_id, df_list in ts_data.items():
        if df_list:
            ts_data[task_id] = pd.concat(df_list, ignore_index=True)
        else:
            ts_data[task_id] = pd.DataFrame()
            
    return ts_data

def main():
    parser = argparse.ArgumentParser(description="Generate plots for Figure 4: Relaxation Dynamics.")
    parser.add_argument("campaign_id", help="Campaign ID for relaxation/perturbation runs.")
    args = parser.parse_args()

    # --- Load Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_path = os.path.join(project_root, "data", args.campaign_id, "analysis", f"{args.campaign_id}_summary_aggregated.csv")
    ts_db_path = os.path.join(project_root, "data", args.campaign_id, "timeseries_raw", f"{args.campaign_id}_timeseries_db.jsonl.gz")

    if not os.path.exists(summary_path) or not os.path.exists(ts_db_path):
        print(f"Error: Required summary or timeseries file not found for campaign {args.campaign_id}"); return
    
    df_summary = pd.read_csv(summary_path)
    print(f"Loaded summary for {len(df_summary)} simulations.")
    
    # --- Process Data: Select a few interesting parameter sets to plot ---
    # Example: Select a few k_total values for a fixed s and phi
    df_summary['s'] = df_summary['b_m'] - 1.0
    phi_to_plot = df_summary['phi'].median()
    s_to_plot = df_summary['s'].median()
    
    df_subset = df_summary[(np.isclose(df_summary['phi'], phi_to_plot)) & (np.isclose(df_summary['s'], s_to_plot))]
    k_values = sorted(df_subset['k_total'].unique())
    k_to_plot = [k_values[0], k_values[len(k_values)//2], k_values[-1]]
    
    task_ids_to_plot = df_subset[df_subset['k_total'].isin(k_to_plot)]['task_id'].tolist()
    
    # Load the actual timeseries for these tasks
    ts_data_map = load_timeseries_data(ts_db_path, set(task_ids_to_plot))
    
    # Combine and average by parameter set
    plot_data = []
    for k in k_to_plot:
        task_ids_for_k = df_subset[df_subset['k_total'] == k]['task_id']
        all_dfs_for_k = [ts_data_map[tid] for tid in task_ids_for_k if not ts_data_map[tid].empty]
        if not all_dfs_for_k: continue
        
        combined_df = pd.concat(all_dfs_for_k)
        # Average across replicates for each time point
        avg_df = combined_df.groupby('time')['mutant_fraction'].mean().reset_index()
        avg_df['k_total'] = k
        plot_data.append(avg_df)

    final_plot_df = pd.concat(plot_data)

    # --- Create Plot ---
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=final_plot_df, x='time', y='mutant_fraction', hue='k_total', palette='viridis', lw=3)
    
    plt.title(f"Relaxation Dynamics (s={s_to_plot:.2f}, phi={phi_to_plot:.2f})")
    plt.xlabel("Time")
    plt.ylabel(r"Mean Mutant Fraction, $\langle\rho_M\rangle$")
    plt.legend(title=r"$k_{total}$", loc="best")
    
    # --- Save Figure ---
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure4_relaxation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved to {output_path}")
    
if __name__ == "__main__":
    main()import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gzip
from tqdm import tqdm

def load_timeseries_data(ts_db_path, task_ids_to_load):
    """Loads specific timeseries from the gzipped JSONL database."""
    ts_data = {task_id: [] for task_id in task_ids_to_load}
    with gzip.open(ts_db_path, 'rt') as f:
        for line in tqdm(f, desc="Loading timeseries DB"):
            try:
                data = json.loads(line)
                task_id = data.get('task_id')
                if task_id in task_ids_to_load:
                    ts_data[task_id].append(pd.DataFrame(data.get('timeseries', [])))
            except json.JSONDecodeError:
                continue
    # Concatenate data for each task_id
    for task_id, df_list in ts_data.items():
        if df_list:
            ts_data[task_id] = pd.concat(df_list, ignore_index=True)
        else:
            ts_data[task_id] = pd.DataFrame()
            
    return ts_data

def main():
    parser = argparse.ArgumentParser(description="Generate plots for Figure 4: Relaxation Dynamics.")
    parser.add_argument("campaign_id", help="Campaign ID for relaxation/perturbation runs.")
    args = parser.parse_args()

    # --- Load Data ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    summary_path = os.path.join(project_root, "data", args.campaign_id, "analysis", f"{args.campaign_id}_summary_aggregated.csv")
    ts_db_path = os.path.join(project_root, "data", args.campaign_id, "timeseries_raw", f"{args.campaign_id}_timeseries_db.jsonl.gz")

    if not os.path.exists(summary_path) or not os.path.exists(ts_db_path):
        print(f"Error: Required summary or timeseries file not found for campaign {args.campaign_id}"); return
    
    df_summary = pd.read_csv(summary_path)
    print(f"Loaded summary for {len(df_summary)} simulations.")
    
    # --- Process Data: Select a few interesting parameter sets to plot ---
    # Example: Select a few k_total values for a fixed s and phi
    df_summary['s'] = df_summary['b_m'] - 1.0
    phi_to_plot = df_summary['phi'].median()
    s_to_plot = df_summary['s'].median()
    
    df_subset = df_summary[(np.isclose(df_summary['phi'], phi_to_plot)) & (np.isclose(df_summary['s'], s_to_plot))]
    k_values = sorted(df_subset['k_total'].unique())
    k_to_plot = [k_values[0], k_values[len(k_values)//2], k_values[-1]]
    
    task_ids_to_plot = df_subset[df_subset['k_total'].isin(k_to_plot)]['task_id'].tolist()
    
    # Load the actual timeseries for these tasks
    ts_data_map = load_timeseries_data(ts_db_path, set(task_ids_to_plot))
    
    # Combine and average by parameter set
    plot_data = []
    for k in k_to_plot:
        task_ids_for_k = df_subset[df_subset['k_total'] == k]['task_id']
        all_dfs_for_k = [ts_data_map[tid] for tid in task_ids_for_k if not ts_data_map[tid].empty]
        if not all_dfs_for_k: continue
        
        combined_df = pd.concat(all_dfs_for_k)
        # Average across replicates for each time point
        avg_df = combined_df.groupby('time')['mutant_fraction'].mean().reset_index()
        avg_df['k_total'] = k
        plot_data.append(avg_df)

    final_plot_df = pd.concat(plot_data)

    # --- Create Plot ---
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 8))
    
    sns.lineplot(data=final_plot_df, x='time', y='mutant_fraction', hue='k_total', palette='viridis', lw=3)
    
    plt.title(f"Relaxation Dynamics (s={s_to_plot:.2f}, phi={phi_to_plot:.2f})")
    plt.xlabel("Time")
    plt.ylabel(r"Mean Mutant Fraction, $\langle\rho_M\rangle$")
    plt.legend(title=r"$k_{total}$", loc="best")
    
    # --- Save Figure ---
    output_dir = os.path.join(project_root, "data", args.campaign_id, "analysis")
    output_path = os.path.join(output_dir, "figure4_relaxation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved to {output_path}")
    
if __name__ == "__main__":
    main()