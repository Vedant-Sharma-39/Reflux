#!/bin/bash
# FILE: scripts/utils/run_analysis_worker.sh
# This script is executed by Slurm on a compute node. It calls the Python
# script in 'worker' mode to process one analysis task.

set -euo pipefail

# 1. ARGUMENT PARSING
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <project_root> <task_list_file> <chunk_output_dir>" >&2
    exit 1
fi
readonly PROJECT_ROOT="$1"
readonly TASK_LIST_FILE="$2"
readonly CHUNK_OUTPUT_DIR="$3"
readonly ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# 2. ENVIRONMENT SETUP
cd "${PROJECT_ROOT}" || exit 1
# Load necessary modules if required by your HPC
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1; module load scipy-stack/2025a
export PROJECT_ROOT

# 3. EXECUTE THE PYTHON WORKER
# The python script will read its task info from the task list file
# based on the SLURM_ARRAY_TASK_ID environment variable.
echo "Running analysis worker for task ${ARRAY_TASK_ID}..."
python3 scripts/utils/process_aif_trajectories.py worker "${TASK_LIST_FILE}" "${CHUNK_OUTPUT_DIR}"

echo "Worker task ${ARRAY_TASK_ID} finished."