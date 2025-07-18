#!/bin/bash
#
# Master launcher for the CALIBRATION campaign.
# Generates calibration tasks and submits the job array to Slurm.

set -eo pipefail

echo "--- Calibration Campaign Launcher ---"
date

# --- Smart Project Root Detection ---
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT=$(pwd)
fi
echo "Project root set to: ${PROJECT_ROOT}"

# --- Get CAMPAIGN_ID from the DEDICATED config file ---
CAMPAIGN_ID=$(python3 -c "import sys, os; sys.path.insert(0, os.path.join('${PROJECT_ROOT}', 'src')); from config_calibration import CAMPAIGN_ID; print(CAMPAIGN_ID)")
if [ -z "${CAMPAIGN_ID}" ]; then
    echo "FATAL: Could not get CAMPAIGN_ID from src/config_calibration.py." >&2
    exit 1
fi
echo "Campaign ID detected: ${CAMPAIGN_ID}"

# --- 1. CONFIGURATION ---
TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/generate_calibration_tasks.py"
RESUME_TASK_FILE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}_task_list.txt"
CHUNK_RUNNER_SCRIPT="${PROJECT_ROOT}/scripts/run_chunk_of_calibration_tasks.sh" # NEW runner
OUTPUT_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results"
LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"

# HPC parameters - these runs might be longer as sectors shrink slowly
SIMS_PER_TASK=100          # Fewer sims per chunk, as each is longer
MAX_CONCURRENT_TASKS=499
CPU_PER_TASK=1
MEM_PER_TASK="4G"
TIME_PER_TASK="0-12:00:00"  # 12 hours, a safe bet for slow shrinkage

# --- 2. GENERATE/RESUME TASK LIST ---
echo ""
echo "Step 1: Generating list of missing calibration tasks..."
python3 "${TASK_GENERATOR_SCRIPT}" --outfile "${RESUME_TASK_FILE}" --clean
if [ $? -ne 0 ]; then
    echo "Task generation script failed. Aborting." >&2
    exit 1
fi

# --- 3. PRE-FLIGHT CHECKS & BATCH CALCULATION ---
if [ ! -s "${RESUME_TASK_FILE}" ]; then
    echo "Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete."
    exit 0
fi

TOTAL_SIMS_TO_RUN=$(wc -l < "${RESUME_TASK_FILE}")
NUM_ARRAY_TASKS=$(( (TOTAL_SIMS_TO_RUN + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo ""
echo "Step 2: Submitting job array..."
echo "   Total simulations to run: ${TOTAL_SIMS_TO_RUN}"
echo   "   Simulations per array task (chunk size): ${SIMS_PER_TASK}"
echo   "   Number of array tasks needed: ${NUM_ARRAY_TASKS}"

# --- 4. SUBMIT THE JOB ARRAY ---
SUBMIT_OUTPUT=$(sbatch \
    --job-name=${CAMPAIGN_ID} \
    --array=1-${NUM_ARRAY_TASKS}%${MAX_CONCURRENT_TASKS} \
    --output=${LOG_DIR}/chunk_task-%A_%a.out \
    --error=${LOG_DIR}/chunk_task-%A_%a.err \
    --cpus-per-task=${CPU_PER_TASK} \
    --mem=${MEM_PER_TASK} \
    --time=${TIME_PER_TASK} \
    "${CHUNK_RUNNER_SCRIPT}" "${RESUME_TASK_FILE}" "${OUTPUT_DIR}" "${PROJECT_ROOT}" "${SIMS_PER_TASK}")

JOB_ID=$(echo "${SUBMIT_OUTPUT}" | awk '{print $4}')

echo ""
echo "==============================================="
echo "Calibration campaign '${CAMPAIGN_ID}' submitted successfully!"
echo "Master Job Array ID: ${JOB_ID}"
echo "Monitor with: squeue -u ${USER} -j ${JOB_ID}"
echo "==============================================="