#!/bin/bash
# FILE: scripts/run_chunk.sh (v4.3 - Corrected Worker Call)
# This version fixes the missing --output-dir argument when calling worker.py.

set -euo pipefail

# 1. ARGUMENT PARSING
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <project_root> <log_dir_base> <task_list_file> <raw_output_dir> <sims_per_chunk>" >&2
    exit 1
fi
readonly PROJECT_ROOT="$1"
readonly TASK_LIST_FILE="$3"
readonly RAW_DATA_DIR="$4"
readonly SIMS_PER_CHUNK="$5"
readonly MAIN_JOB_ID=${SLURM_ARRAY_JOB_ID}
readonly ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# 2. ENVIRONMENT SETUP
cd "${PROJECT_ROOT}" || exit 1
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi
module purge >/dev/null 2>&1; module load scipy-stack/2024a
export PROJECT_ROOT

# 3. MAIN LOGIC
echo "======================================================"
echo "Slurm Job           : ${MAIN_JOB_ID}_${ARRAY_TASK_ID}"
echo "Running on Node     : $(hostname)"
echo "Sims per this chunk : ${SIMS_PER_CHUNK}"
echo "------------------------------------------------------"

readonly OUTPUT_FILE="${RAW_DATA_DIR}/chunk_${MAIN_JOB_ID}_${ARRAY_TASK_ID}.jsonl"
touch "${OUTPUT_FILE}"

readonly START_LINE=$(( (ARRAY_TASK_ID - 1) * SIMS_PER_CHUNK + 1 ))
readonly END_LINE=$(( ARRAY_TASK_ID * SIMS_PER_CHUNK ))

echo "INFO: Processing lines ${START_LINE}-${END_LINE} from ${TASK_LIST_FILE##*/}"
echo "INFO: Appending results to ${OUTPUT_FILE##*/}"

num_processed=0
num_failed=0

while IFS= read -r PARAMS_JSON; do
    if [ -z "${PARAMS_JSON}" ]; then continue; fi
    task_id=$(python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))" <<< "${PARAMS_JSON}")
    echo -n "[$(date '+%T')] Processing task ID: ${task_id}... "
    
    # --- THE CRITICAL FIX ---
    # The worker script must be called with both --params and --output-dir arguments.
    # Its stdout (the summary) is redirected to the chunk file.
    # Its stderr is redirected to the main slurm log for debugging.
    if ! python3 "src/worker.py" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}" >> "${OUTPUT_FILE}"; then
        # The worker automatically prints a detailed error to stderr, which is captured by Slurm.
        echo "FAILED. See this log for the worker's error message."
        ((num_failed++))
    else
        echo "OK."
    fi
    # --- END FIX ---

    ((num_processed++))
done < <(sed -n "${START_LINE},${END_LINE}p" "${TASK_LIST_FILE}")

echo -e "\nINFO: Chunk finished. Attempted: ${num_processed}, Failed: ${num_failed}"
if [ "$num_failed" -gt 0 ]; then exit 1; fi