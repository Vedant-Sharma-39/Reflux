#!/bin/bash
# FILE: scripts/run_chunk.sh (Corrected with -m flag to fix ModuleNotFoundError)
# This version fixes the subtle but critical bug where `set -e` would cause
# the script to exit prematurely when the `while read` loop finished.

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
if [ -f /etc/profile/d/modules.sh ]; then
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

# Use process substitution to feed lines from sed into the while loop
# The '|| true' at the end is the critical fix. It prevents 'set -e'
# from killing the script when the while loop finishes reading its input.
while IFS= read -r PARAMS_JSON; do
    if [ -z "${PARAMS_JSON}" ]; then continue; fi
    
    # Use a "here string" (<<<) to prevent input hijacking.
    task_id=$(python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))" <<< "${PARAMS_JSON}")

    echo -n "[$(date '+%T')] Processing task ID: ${task_id}... "
    
    # --- THE FIX IS HERE ---
    # We now run the worker as a module (-m) from the project root.
    # This ensures that Python's import system can find the 'src' package.
    if ! python3 -m "src.worker" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}" >> "${OUTPUT_FILE}"; then
        echo "FAILED. See this log for the worker's error message."
        ((num_failed++))
    else
        echo "OK."
    fi
    # --- END OF FIX ---

    ((num_processed++))
done < <(sed -n "${START_LINE},${END_LINE}p" "${TASK_LIST_FILE}") || true

echo -e "\nINFO: Chunk finished. Attempted: ${num_processed}, Failed: ${num_failed}"
if [ "$num_failed" -gt 0 ]; then exit 1; fi