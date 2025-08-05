#!/bin/bash
# FILE: scripts/launch.sh
# [DEFINITIVE VERSION]
# Manages the simulation lifecycle. This version correctly uses a dependency-free
# config and handles the modern data pipeline.

set -eo pipefail

# --- Style and Helper Functions ---
readonly C_RED="\033[1;31m"; readonly C_GREEN="\033[1;32m"; readonly C_BLUE="\033[1;34m"; readonly C_YELLOW="\033[1;33m"; readonly C_NC="\033[0m"

confirm_action() {
    local prompt="$1"
    read -p "$(echo -e ${C_YELLOW}"${prompt} [y/N]: "${C_NC})" -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then echo -e "${C_RED}Aborted by user.${C_NC}"; exit 1; fi
}

if [ -n "${SLURM_SUBMIT_DIR}" ]; then readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"; else readonly PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || realpath "$(dirname "$0")/../.."); fi

# --- Argument Parsing and UI ---
readonly ACTIONS=("launch" "status" "consolidate" "clean" "debug-task")
mapfile -t EXPERIMENT_NAMES < <(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print('\n'.join(EXPERIMENTS.keys()))" 2>/dev/null || echo "ERROR: Could not load experiments")
if [[ "${EXPERIMENT_NAMES[0]}" == "ERROR:"* ]]; then echo -e "${C_RED}${EXPERIMENT_NAMES[0]}\nPlease check your src/config.py for syntax errors.${C_NC}"; exit 1; fi

ACTION="${1:-}"; EXPERIMENT_NAME="${2:-}"
if [ -z "$ACTION" ]; then
    echo -e "${C_BLUE}Choose an action:${C_NC}"; PS3="Action: "; select CHOICE in "${ACTIONS[@]}"; do [[ -n "$CHOICE" ]] && { ACTION="$CHOICE"; break; }; done
fi
if [ -z "$EXPERIMENT_NAME" ]; then
    echo -e "${C_BLUE}Choose an experiment:${C_NC}"; PS3="Experiment: "; select CHOICE in "${EXPERIMENT_NAMES[@]}"; do [[ -n "$CHOICE" ]] && { EXPERIMENT_NAME="$CHOICE"; break; }; done
fi; echo

# --- Load Configuration Variables (Robustly) ---
get_config_values() {
    # [FIXED] This python script now prints a single, space-separated line
    # to be compatible with the shell's `read` command.
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/src')
from config import EXPERIMENTS
e = EXPERIMENTS['${EXPERIMENT_NAME}']
h = e.get('HPC_PARAMS', {})
campaign_id = e.get('CAMPAIGN_ID', 'default_campaign')
sims_per_task = h.get('sims_per_task', 100)
mem = h.get('mem', '2G')
time = h.get('time', '01:00:00')
print(f'{campaign_id} {sims_per_task} {mem} {time}')
"
}
read -r CAMPAIGN_ID SIMS_PER_TASK MEM_PER_TASK TIME_PER_TASK < <(get_config_values)

# --- Define Paths ---
readonly TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/utils/generate_tasks.py"
readonly AGGREGATOR_SCRIPT="${PROJECT_ROOT}/scripts/utils/aggregate_data.py"
readonly DATA_DIR_BASE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"
readonly RESUME_TASK_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_resume_tasks.txt"
readonly TOTAL_TASK_COUNT_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_total_tasks.txt"

echo -e "${C_GREEN}Action: ${C_YELLOW}${ACTION}${C_GREEN} | Campaign: ${C_YELLOW}${CAMPAIGN_ID}${C_NC}"
echo "-----------------------------------------------------"

case "$ACTION" in
    "launch"|"status")
        echo "Step 1: Consolidating any completed raw data..."
        python3 "${AGGREGATOR_SCRIPT}" "${CAMPAIGN_ID}"
        echo "Step 2: Generating/updating task lists..."
        python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}"
        readonly MASTER_SUMMARY_FILE="${DATA_DIR_BASE}/analysis/${CAMPAIGN_ID}_summary_aggregated.csv"
        COMPLETED_TASK_COUNT=0
        if [ -f "${MASTER_SUMMARY_FILE}" ]; then COMPLETED_TASK_COUNT=$(awk 'NR>1' "${MASTER_SUMMARY_FILE}" | wc -l); fi
        if [ ! -f "${TOTAL_TASK_COUNT_FILE}" ]; then echo -e "${C_RED}Error: Total task count file not found.${C_NC}" >&2; exit 1; fi
        readonly TOTAL_UNIVERSE=$(cat "${TOTAL_TASK_COUNT_FILE}")
        PERCENTAGE=$(awk "BEGIN {if ($TOTAL_UNIVERSE > 0) printf \"%.2f\", $COMPLETED_TASK_COUNT / $TOTAL_UNIVERSE * 100; else print \"0.00\"}")
        echo -e "\nCampaign Progress: ${C_GREEN}${COMPLETED_TASK_COUNT} / ${TOTAL_UNIVERSE} tasks complete (${PERCENTAGE}%)${C_NC}"
        if [ "$ACTION" == "status" ]; then exit 0; fi
        if [ ! -s "${RESUME_TASK_FILE}" ]; then echo -e "${C_GREEN}Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete.${C_NC}"; exit 0; fi
        if ! [[ "${SIMS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then echo -e "${C_RED}Error: 'sims_per_task' is not a positive integer (value: '${SIMS_PER_TASK}').${C_NC}" >&2; exit 1; fi
        readonly TOTAL_SIMS_TO_RUN=$(wc -l < "${RESUME_TASK_FILE}")
        readonly NUM_ARRAY_TASKS=$(( (TOTAL_SIMS_TO_RUN + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
        echo -e "\nAbout to submit ${C_YELLOW}${NUM_ARRAY_TASKS}${C_NC} array jobs to Slurm."
        echo "Config: ${SIMS_PER_TASK} sims/task, ${MEM_PER_TASK} memory, ${TIME_PER_TASK} time."
        confirm_action "Proceed with submission?"
        readonly LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"; mkdir -p "${LOG_DIR}"
        sbatch --job-name="${CAMPAIGN_ID}" --array=1-"${NUM_ARRAY_TASKS}"%500 --output="${LOG_DIR}/chunk_task-%A_%a.log" --cpus-per-task=1 --mem="${MEM_PER_TASK}" --time="${TIME_PER_TASK}" "${PROJECT_ROOT}/scripts/run_chunk.sh" "${RESUME_TASK_FILE}" "${DATA_DIR_BASE}" "${PROJECT_ROOT}" "${SIMS_PER_TASK}"
        echo -e "\n${C_GREEN}Campaign '${CAMPAIGN_ID}' submitted successfully!${C_NC}"
        ;;
    "consolidate") python3 "${AGGREGATOR_SCRIPT}" "${CAMPAIGN_ID}";;
    "clean") confirm_action "Are you sure you want to DELETE ALL data for ${CAMPAIGN_ID}?"; rm -rf "${DATA_DIR_BASE}" "${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"; echo "Cleanup complete.";;
    "debug-task")
        if [ ! -f "${RESUME_TASK_FILE}" ]; then echo "Task file not found. Generating it first..."; python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}"; fi
        read -p "Enter task line number to debug from ${RESUME_TASK_FILE}: " LINE_NUM
        PARAMS_JSON=$(sed -n "${LINE_NUM}p" "${RESUME_TASK_FILE}")
        echo "--- Running worker with parameters: ---"; echo "${PARAMS_JSON}" | python3 -m json.tool
        echo "--- Worker Output: ---"; python3 "${PROJECT_ROOT}/src/worker.py" --params "${PARAMS_JSON}"
        ;;
    *) echo -e "${C_RED}Invalid action: $ACTION. Valid actions are: ${ACTIONS[*]}${C_NC}" >&2; exit 1;;
esac