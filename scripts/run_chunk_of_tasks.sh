#!/bin/bash
# [CORRECTED] This script is executed by each Slurm task for the phase transition campaign.
# Its logic is now identical to the working calibration chunk runner.

set -euo pipefail

# --- 1. Validate Input Arguments ---
if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments." >&2
    exit 1
fi
TASK_FILE="$1"
OUTPUT_DIR="$2"
PROJECT_ROOT="$3"
SIMS_PER_TASK="$4"

echo "--- Phase Worker Chunk ${SLURM_ARRAY_TASK_ID} Started on $(hostname) ---"

# --- 2. Calculate Simulation Range ---
START_SIM_NUM=$(( (SLURM_ARRAY_TASK_ID - 1) * SIMS_PER_TASK + 1 ))
END_SIM_NUM=$(( SLURM_ARRAY_TASK_ID * SIMS_PER_TASK ))
echo "Processing lines ${START_SIM_NUM} to ${END_SIM_NUM} from task file."

# --- 3. Set up Environment ---
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"
# Point to the correct worker for this campaign
WORKER_SCRIPT="${PROJECT_ROOT}/src/phase_worker.py"
cd "${PROJECT_ROOT}" || exit 1

TOTAL_LINES_IN_FILE=$(wc -l < "${TASK_FILE}")

# --- 4. Main Execution Loop ---
for (( SIM_NUM=START_SIM_NUM; SIM_NUM<=END_SIM_NUM; SIM_NUM++ )); do
    if [ ${SIM_NUM} -gt ${TOTAL_LINES_IN_FILE} ]; then
        echo "Reached end of task file. Worker exiting."
        break
    fi

    PARAMS_JSON=$(sed -n "${SIM_NUM}p" "${TASK_FILE}" || true)
    if [ -z "${PARAMS_JSON}" ]; then
        continue
    fi
    
    # Check for existing result to make the chunk resumable
    TASK_ID_PRECHECK=$(echo "${PARAMS_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))")
    OUTPUT_FILE="${OUTPUT_DIR}/result_${TASK_ID_PRECHECK}.json"
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "Result for task line ${SIM_NUM} (ID: ${TASK_ID_PRECHECK}) exists. Skipping."
        continue
    fi

    echo "Running phase transition sim for line ${SIM_NUM}..."
    RESULT_JSON=$(python3 "${WORKER_SCRIPT}" --params "${PARAMS_JSON}")
    
    if [ -z "${RESULT_JSON}" ]; then
        echo "Warning: Python worker for line ${SIM_NUM} produced no output. Skipping."
        continue
    fi

    # This is the two-step saving logic from your working script.
    # It parses the task_id from the *output* of the worker.
    TASK_ID=$(echo "${RESULT_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin).get('task_id', 'unknown'))")
    SAVED_OUTPUT_FILE="${OUTPUT_DIR}/result_${TASK_ID}.json"
    echo "${RESULT_JSON}" > "${SAVED_OUTPUT_FILE}"
done

echo "--- Phase Worker Chunk ${SLURM_ARRAY_TASK_ID} Finished ---"