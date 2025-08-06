#!/bin/bash
# FILE: scripts/run_chunk.sh
# [v_FINAL_HYBRID] Processes a "chunk" of tasks from a given run list.
# This script is called by Slurm for each array job.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <run_list_file> <raw_output_dir> <sims_per_chunk>" >&2
    exit 1
fi

readonly RUN_LIST_FILE="$1"
readonly RAW_OUTPUT_DIR="$2"
readonly SIMS_PER_CHUNK="$3"

# PROJECT_ROOT is inherited from the submission environment (set by launch.sh)
if [ -z "${PROJECT_ROOT}" ]; then echo "ERROR: PROJECT_ROOT not set." >&2; exit 1; fi

# Slurm Environment
readonly SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

# Calculate the line numbers this chunk is responsible for
readonly START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * SIMS_PER_CHUNK + 1 ))
readonly END_LINE=$(( SLURM_ARRAY_TASK_ID * SIMS_PER_CHUNK ))

# Use sed to efficiently extract the lines for this chunk
# This is much faster than a bash loop for large files
sed -n "${START_LINE},${END_LINE}p" "${RUN_LIST_FILE}" | while IFS= read -r PARAMS_JSON; do
    if [ -z "${PARAMS_JSON}" ]; then continue; fi

    # Use a subshell to isolate each worker run and prevent script exit on a single failure
    (
      echo "Processing task..."
      python3 "${PROJECT_ROOT}/src/worker.py" --params "${PARAMS_JSON}" --output-dir "${RAW_OUTPUT_DIR}"
    ) || {
      # This block captures python exit codes != 0
      echo "ERROR: Worker failed for task defined by:"
      echo "${PARAMS_JSON}"
      # The worker now creates a .error file, so we don't need to do more here.
    }
done

echo "Chunk ${SLURM_ARRAY_TASK_ID} finished."