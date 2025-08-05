#!/bin/bash
# FILE: scripts/run_chunk.sh
# [v3 - FILE-FRUGAL, RAW OUTPUT]
# This script runs a chunk of simulations and writes output to temporary
# chunk files in a dedicated 'raw' directory.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Error: Incorrect number of arguments." >&2
    echo "Usage: $0 <task_file> <output_dir_base> <project_root> <sims_per_task>"
    exit 1
fi
TASK_FILE="$1"
OUTPUT_DIR_BASE="$2"
PROJECT_ROOT="$3"
SIMS_PER_TASK="$4"

# --- ### NEW: Define temporary RAW output directories and files ### ---
CAMPAIGN_ID=$(basename "$OUTPUT_DIR_BASE")
RAW_RESULTS_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results_raw"
RAW_TIMESERIES_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/timeseries_raw"
mkdir -p "${RAW_RESULTS_DIR}" "${RAW_TIMESERIES_DIR}"

CHUNK_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SUMMARY_CHUNK_FILE="${RAW_RESULTS_DIR}/results_chunk_${CHUNK_ID}.jsonl"
TIMESERIES_CHUNK_FILE="${RAW_TIMESERIES_DIR}/ts_chunk_${CHUNK_ID}.jsonl"

echo "--- Worker Chunk ${SLURM_ARRAY_TASK_ID} Started ---"
echo "Writing raw summary to: ${SUMMARY_CHUNK_FILE}"

START_SIM_NUM=$(( (SLURM_ARRAY_TASK_ID - 1) * SIMS_PER_TASK + 1 ))
END_SIM_NUM=$(( SLURM_ARRAY_TASK_ID * SIMS_PER_TASK ))

export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}/src"
WORKER_SCRIPT="${PROJECT_ROOT}/src/worker.py"
cd "${PROJECT_ROOT}" || exit 1

TOTAL_LINES_IN_FILE=$(wc -l < "${TASK_FILE}")
DELIMITER="---WORKER_PAYLOAD_SEPARATOR---"

for (( SIM_NUM=START_SIM_NUM; SIM_NUM<=END_SIM_NUM; SIM_NUM++ )); do
    if [ ${SIM_NUM} -gt ${TOTAL_LINES_IN_FILE} ]; then
        echo "Reached end of task file."
        break
    fi
    PARAMS_JSON=$(sed -n "${SIM_NUM}p" "${TASK_FILE}" || true)
    if [ -z "${PARAMS_JSON}" ]; then
        continue
    fi
    
    WORKER_OUTPUT=$(python3 "${WORKER_SCRIPT}" --params "${PARAMS_JSON}" || true)
    if [ -z "${WORKER_OUTPUT}" ]; then
        echo "Warning: Python worker for line ${SIM_NUM} produced no output."
        continue
    fi
    
    SUMMARY_JSON=$(echo "${WORKER_OUTPUT}" | awk -v RS="${DELIMITER}" 'NR==1')
    TIMESERIES_JSON=$(echo "${WORKER_OUTPUT}" | awk -v RS="${DELIMITER}" 'NR==2')

    echo "${SUMMARY_JSON}" >> "${SUMMARY_CHUNK_FILE}"
    
    # Only write timeseries if it's not empty
    if [[ $(echo "${TIMESERIES_JSON}" | jq '.timeseries | length') -gt 0 ]]; then
        echo "${TIMESERIES_JSON}" >> "${TIMESERIES_CHUNK_FILE}"
    fi
done

# Compress timeseries chunk if it was created
if [ -f "${TIMESERIES_CHUNK_FILE}" ]; then
    gzip "${TIMESERIES_CHUNK_FILE}"
fi

echo "--- Worker Chunk ${SLURM_ARRAY_TASK_ID} Finished ---"