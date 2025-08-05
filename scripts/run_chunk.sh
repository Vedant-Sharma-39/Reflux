#!/bin/bash
# FILE: scripts/run_chunk.sh
# [DEFINITIVE VERSION]
# This script runs a chunk of simulations. It is idempotent and uses efficient
# methods for processing worker output.

set -euo pipefail

if [ "$#" -ne 4 ]; then echo "Usage: $0 <task_file> <data_dir_base> <project_root> <sims_per_task>" >&2; exit 1; fi

readonly TASK_FILE="$1"; readonly DATA_DIR_BASE="$2"; readonly PROJECT_ROOT="$3"; readonly SIMS_PER_TASK="$4"
readonly WORKER_SCRIPT="${PROJECT_ROOT}/src/worker.py"; readonly DELIMITER="---WORKER_PAYLOAD_SEPARATOR---"

readonly SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-local}; readonly SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
readonly CAMPAIGN_ID=$(basename "$DATA_DIR_BASE")
readonly RAW_RESULTS_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results_raw"
readonly RAW_TIMESERIES_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/timeseries_raw"
mkdir -p "${RAW_RESULTS_DIR}" "${RAW_TIMESERIES_DIR}"

readonly CHUNK_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly SUMMARY_CHUNK_FILE="${RAW_RESULTS_DIR}/results_chunk_${CHUNK_ID}.jsonl"
readonly TIMESERIES_CHUNK_FILE="${RAW_TIMESERIES_DIR}/ts_chunk_${CHUNK_ID}.jsonl"

rm -f "${SUMMARY_CHUNK_FILE}" "${TIMESERIES_CHUNK_FILE}" "${TIMESERIES_CHUNK_FILE}.gz"
echo "--- Worker Chunk ${SLURM_ARRAY_TASK_ID} Started for Campaign: ${CAMPAIGN_ID} ---"

readonly START_SIM_NUM=$(( (SLURM_ARRAY_TASK_ID - 1) * SIMS_PER_TASK + 1 ))
readonly END_SIM_NUM=$(( SLURM_ARRAY_TASK_ID * SIMS_PER_TASK ))
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}"
cd "${PROJECT_ROOT}" || exit 1
readonly TOTAL_LINES_IN_FILE=$(wc -l < "${TASK_FILE}")

for (( SIM_NUM=START_SIM_NUM; SIM_NUM<=END_SIM_NUM; SIM_NUM++ )); do
    if [ ${SIM_NUM} -gt ${TOTAL_LINES_IN_FILE} ]; then echo "Reached end of task file. Stopping."; break; fi
    PARAMS_JSON=$(sed -n "${SIM_NUM}p" "${TASK_FILE}" || true)
    if [ -z "${PARAMS_JSON}" ]; then echo "Warning: Line ${SIM_NUM} is empty. Skipping."; continue; fi

    { read -r SUMMARY_JSON; read -r TIMESERIES_JSON; } < <(python3 "${WORKER_SCRIPT}" --params "${PARAMS_JSON}" | sed "s/${DELIMITER}/\n/" || true)

    if [ -z "${SUMMARY_JSON}" ]; then echo "Warning: Worker for line ${SIM_NUM} produced no output."; continue; fi
    echo "${SUMMARY_JSON}" >> "${SUMMARY_CHUNK_FILE}"
    if [[ ${TIMESERIES_JSON} != *'"timeseries":[]'* ]]; then echo "${TIMESERIES_JSON}" >> "${TIMESERIES_CHUNK_FILE}"; fi
done

if [ -s "${TIMESeries_CHUNK_FILE}" ]; then gzip "${TIMESERIES_CHUNK_FILE}"; echo "Compressed timeseries chunk."; fi
echo "--- Worker Chunk ${SLURM_ARRAY_TASK_ID} Finished ---"