#!/bin/bash
# FILE: scripts/run_chunk.sh (v2.0 - With Robust Logging)
# This script is executed by each Slurm array task. It is self-sufficient,
# sets up its own environment, and handles its own log file management.

set -euo pipefail

# ==============================================================================
# [CRITICAL] 1. SCRIPT SETUP and ARGUMENT PARSING
# ==============================================================================
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <job_log_dir> <task_list_file> <raw_output_dir> <sims_per_chunk>" >&2
    exit 1
fi

readonly JOB_LOG_DIR="$1"
readonly TASK_LIST_FILE="$2"
readonly RAW_DATA_DIR="$3"
readonly SIMS_PER_CHUNK="$4"

# Slurm Environment Variables
readonly MAIN_JOB_ID=${SLURM_ARRAY_JOB_ID}
readonly ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Define log file paths
readonly LOG_FILE_RUNNING="${JOB_LOG_DIR}/running/task_${ARRAY_TASK_ID}.log"
readonly LOG_FILE_SUCCESS="${JOB_LOG_DIR}/success/task_${ARRAY_TASK_ID}.log"
readonly LOG_FILE_FAILED="${JOB_LOG_DIR}/failed/task_${ARRAY_TASK_ID}.log"

# ==============================================================================
# [CRITICAL] 2. ERROR HANDLING with trap
# ==============================================================================
# This function will be executed automatically when the script exits, for any reason.
cleanup() {
    local exit_code=$? # Get the exit code of the last command
    
    # Add a final summary to the log file
    echo "------------------------------------------------------" >> "${LOG_FILE_RUNNING}"
    if [ ${exit_code} -eq 0 ]; then
        echo "STATUS: SUCCESS" >> "${LOG_FILE_RUNNING}"
        # Move the log file to the 'success' directory
        mv "${LOG_FILE_RUNNING}" "${LOG_FILE_SUCCESS}"
    else
        echo "STATUS: FAILED (Exit Code: ${exit_code})" >> "${LOG_FILE_RUNNING}"
        echo "ERROR: A command failed. Check the output above." >> "${LOG_FILE_RUNNING}"
        # Move the log file to the 'failed' directory
        mv "${LOG_FILE_RUNNING}" "${LOG_FILE_FAILED}"
    fi
}
# Register the 'cleanup' function to run on script EXIT
trap cleanup EXIT

# ==============================================================================
# [CRITICAL] 3. ENVIRONMENT SETUP
# ==============================================================================
# This section MUST be present. It ensures the job runs with the correct software.
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1
module load scipy-stack/2024a

# --- Find project root relative to this script ---
readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

# ==============================================================================
# [CRITICAL] 4. MAIN LOGIC
# ==============================================================================

# --- Start of Log Content ---
echo "======================================================"
echo "Slurm Job           : ${MAIN_JOB_ID}_${ARRAY_TASK_ID}"
echo "Running on Node     : $(hostname)"
echo "Start Time          : $(date)"
echo "------------------------------------------------------"

# --- Calculate task range ---
readonly START_LINE=$(( (ARRAY_TASK_ID - 1) * SIMS_PER_CHUNK + 1 ))
readonly END_LINE=$(( ARRAY_TASK_ID * SIMS_PER_CHUNK ))

echo "INFO: Processing lines ${START_LINE}-${END_LINE} from ${TASK_LIST_FILE##*/}"
echo ""

# --- Process the assigned chunk of tasks ---
num_processed=0
(
    cd "${PROJECT_ROOT}" && \
    sed -n "${START_LINE},${END_LINE}p" "${TASK_LIST_FILE}" | while IFS= read -r PARAMS_JSON; do
        if [ -z "${PARAMS_JSON}" ]; then continue; fi
        
        task_id=$(echo "${PARAMS_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))")
        echo "[$(date '+%T')] Processing task ID: ${task_id}..."
        
        # Execute the worker and capture its output directly into this log file.
        # The worker's own logic will handle skipping completed tasks.
        python3 "src/worker.py" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}"
        
        ((num_processed++))
    done
)

echo ""
echo "INFO: Finished processing. ${num_processed} tasks attempted in this chunk."
# The 'trap' will handle the rest upon successful exit.