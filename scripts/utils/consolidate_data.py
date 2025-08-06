# FILE: scripts/utils/consolidate_data.py
import pandas as pd
import json, os, sys

campaign_id = sys.argv[1]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
raw_dir = os.path.join(project_root, "data", campaign_id, "raw")
analysis_dir = os.path.join(project_root, "data", campaign_id, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

all_results = []
for fname in os.listdir(raw_dir):
    if fname.endswith(".json"):
        with open(os.path.join(raw_dir, fname), "r") as f:
            # The raw file might contain the result dict, while the params are in the master list
            # It's often better to re-load the params and merge with results.
            # For simplicity here, let's assume the worker saves everything.
            all_results.append(json.load(f))

if not all_results:
    print("No raw results found to consolidate.")
    exit(0)

df = pd.DataFrame(all_results)
output_file = os.path.join(analysis_dir, f"{campaign_id}_summary.csv")
df.to_csv(output_file, index=False)
print(f"Consolidated {len(df)} results into {output_file}")
