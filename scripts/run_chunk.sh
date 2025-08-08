#!/bin/bash
# FILE: scripts/run_chunk.sh (v2.7 - Final, Exit Code Fix)
# This version fixes a subtle bug where the 'while read' loop's natural
# termination on EOF (which has a non-zero exit code) was being
# incorrectly interpreted by the exit trap as a script failure.

set -euo pipefail

# ==============================================================================
# 1. SCRIPT SETUP and ARGUMENT PARSING
# ==============================================================================
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <project_root> <job_log_dir_base> <task_list_file> <raw_output_dir> <sims_per_chunk>" >&2
    exit 1
fi

readonly PROJECT_ROOT="$1"
readonly BASE_LOG_DIR="$2"
readonly TASK_LIST_FILE="$3"
readonly RAW_DATA_DIR="$4"
readonly SIMS_PER_CHUNK="$5"

# --- Slurm Environment Variables ---
readonly MAIN_JOB_ID=${SLURM_ARRAY_JOB_ID}
readonly ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Reconstruct the true log directory path
readonly JOB_LOG_DIR="${BASE_LOG_DIR}/${MAIN_JOB_ID}"

# Define log file paths
readonly LOG_FILE_RUNNING="${JOB_LOG_DIR}/running/task_${ARRAY_TASK_ID}.log"
readonly LOG_FILE_SUCCESS="${JOB_LOG_DIR}/success/task_${ARRAY_TASK_ID}.log"
readonly LOG_FILE_FAILED="${JOB_LOG_DIR}/failed/task_${ARRAY_TASK_ID}.log"

# ==============================================================================
# 2. ERROR HANDLING with trap
# ==============================================================================
cleanup() {
    local exit_code=$?
    echo "------------------------------------------------------" >> "${LOG_FILE_RUNNING}"
    if [ ${exit_code} -eq 0 ]; then
        echo "STATUS: SUCCESS" >> "${LOG_FILE_RUNNING}"
        mv "${LOG_FILE_RUNNING}" "${LOG_FILE_SUCCESS}"
    else
        echo "STATUS: FAILED (Exit Code: ${exit_code})" >> "${LOG_FILE_RUNNING}"
        echo "ERROR: A command failed. Check the output above." >> "${LOG_FILE_RUNNING}"
        mv "${LOG_FILE_RUNNING}" "${LOG_FILE_FAILED}"
    fi
}
trap cleanup EXIT

# ==============================================================================
# 3. ENVIRONMENT SETUP
# ==============================================================================
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1
module load scipy-stack/2024a

export PROJECT_ROOT

# ==============================================================================
# 4. MAIN LOGIC
# ==============================================================================
echo "======================================================"
echo "Slurm Job           : ${MAIN_JOB_ID}_${ARRAY_TASK_ID}"
echo "Running on Node     : $(hostname)"
echo "Start Time          : $(date)"
echo "Project Root (Recv) : ${PROJECT_ROOT}"

cd "${PROJECT_ROOT}" || { echo "FATAL: Could not cd to ${PROJECT_ROOT}"; exit 1; }
echo "Current Directory   : $(pwd)"
echo "------------------------------------------------------"

# --- Calculate task range ---
readonly START_LINE=$(( (ARRAY_TASK_ID - 1) * SIMS_PER_CHUNK + 1 ))
readonly END_LINE=$(( ARRAY_TASK_ID * SIMS_PER_CHUNK ))

echo "INFO: Processing lines ${START_LINE}-${END_LINE} from ${TASK_LIST_FILE##*/}"
echo ""

num_processed=0

while IFS= read -r PARAMS_JSON; do
    if [ -z "${PARAMS_JSON}" ]; then continue; fi
    
    task_id=$(echo "${PARAMS_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))")
    echo "[$(date '+%T')] Processing task ID: ${task_id}..."
    
    python3 "src/worker.py" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}"
    
    ((num_processed++))
done < <(sed -n "${START_LINE},${END_LINE}p" "${TASK_LIST_FILE}") || true


echo ""
echo "INFO: Finished processing. ${num_processed} tasks attempted in this chunk."