#!/bin/bash
#
# This script is the master launcher for the Phase 1 campaign.
# It can be run directly from the command line on the login node.
# It will generate the necessary tasks and then submit the actual
# job array to the Slurm scheduler.

# Use -e to exit on error, -o pipefail to handle pipe failures.
set -eo pipefail

echo "--- Ultimate Campaign Launcher ---"
date

# --- Smart Project Root Detection ---
# This allows the script to be run from the project root directory directly
# or when submitted to Slurm (if you were to wrap this in another sbatch).
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
    echo "Running under Slurm job. Project root set from SLURM_SUBMIT_DIR: ${PROJECT_ROOT}"
else
    PROJECT_ROOT=$(pwd)
    echo "Running interactively. Project root set from current directory: ${PROJECT_ROOT}"
fi

# --- Get CAMPAIGN_ID from src/config.py ---
# This is crucial for dynamically setting paths based on the current campaign.
# Uses a Python one-liner to safely extract the CAMPAIGN_ID.
CAMPAIGN_ID=$(python3 -c "import sys, os; sys.path.insert(0, os.path.join('${PROJECT_ROOT}', 'src')); from config import CAMPAIGN_ID; print(CAMPAIGN_ID)")
if [ -z "${CAMPAIGN_ID}" ]; then
    echo "FATAL: Could not retrieve CAMPAIGN_ID from src/config.py. Exiting." >&2
    exit 1
fi
echo "Campaign ID detected: ${CAMPAIGN_ID}"

# --- 1. CONFIGURATION ---
TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/generate_p1_definitive_tasks.py"
# Paths are now dynamic based on CAMPAIGN_ID
RESUME_TASK_FILE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}_task_list.txt"
CHUNK_RUNNER_SCRIPT="${PROJECT_ROOT}/scripts/run_chunk_of_tasks.sh"
OUTPUT_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results"
LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}" # Logs now organized per campaign

# HPC parameters for each chunk job (adjust based on profiling single chunk runs)
SIMS_PER_TASK=100           # How many short simulations to bundle into one Slurm job.
MAX_CONCURRENT_TASKS=499    # Max concurrent array tasks.
CPU_PER_TASK=1              # Each simulation is single-threaded.
MEM_PER_TASK="4G"           # Generous memory for 100 Python simulations.
TIME_PER_TASK="0-06:00:00"  # 6 hours: Estimate based on 100 * ~3-4 minutes per longest sim.

# --- 2. GENERATE/RESUME TASK LIST ---
echo ""
echo "Step 1: Generating list of missing tasks..."
# Call the Python generator with --clean flag to remove old/invalid result files
# that do not belong to the current campaign ID. This is critical for data integrity.
python3 "${TASK_GENERATOR_SCRIPT}" --outfile "${RESUME_TASK_FILE}" --clean
if [ $? -ne 0 ]; then
    echo "Task generation script failed. Aborting." >&2
    exit 1
fi

# --- 3. PRE-FLIGHT CHECKS & BATCH CALCULATION ---
# Check if the task list file exists and is not empty.
# `-s` checks if file exists and has a size greater than zero.
if [ ! -s "${RESUME_TASK_FILE}" ]; then
    echo "Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete. Nothing to submit."
    exit 0
fi

TOTAL_SIMS_TO_RUN=$(wc -l < "${RESUME_TASK_FILE}")
# Calculate number of array tasks needed using ceiling division.
NUM_ARRAY_TASKS=$(( (TOTAL_SIMS_TO_RUN + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo ""
echo "Step 2: Submitting job array..."
echo "   Campaign ID: ${CAMPAIGN_ID}"
echo "   Total simulations to run in this campaign: ${TOTAL_SIMS_TO_RUN}"
echo "   Simulations per array task (chunk size): ${SIMS_PER_TASK}"
echo "   Calculated number of array tasks needed: ${NUM_ARRAY_TASKS}"
echo "   Estimated resources per array task: ${CPU_PER_TASK} CPU, ${MEM_PER_TASK} RAM, ${TIME_PER_TASK} wall time."

# --- 4. SUBMIT THE JOB ARRAY ---
# We now submit the CHUNK_RUNNER_SCRIPT (which runs many sims serially), not this manager script.
SUBMIT_OUTPUT=$(sbatch \
    --job-name=${CAMPAIGN_ID}_workers \
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
echo "Definitive campaign '${CAMPAIGN_ID}' submitted successfully!"
echo "Master Job Array ID: ${JOB_ID}"
echo "Monitor with: squeue -u ${USER} -j ${JOB_ID}"
echo "==============================================="