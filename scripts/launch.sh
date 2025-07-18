#!/bin/bash
# FILE: scripts/launch.sh
# Campaign Management CLI for the Reflux Project.
# Provides actions to launch, check status, clean, and debug campaigns.

set -eo pipefail

# --- BASH HELPER FUNCTIONS ---
# Add some color to the output
C_RED="\033[1;31m"
C_GREEN="\033[1;32m"
C_BLUE="\033[1;34m"
C_YELLOW="\033[1;33m"
C_NC="\033[0m" # No Color

# Function to ask for user confirmation
confirm_action() {
    local prompt="$1"
    read -p "$(echo -e ${C_YELLOW}"${prompt} [y/N]: "${C_NC})" -n 1 -r
    echo # Move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${C_RED}Aborted.${C_NC}"
        exit 1
    fi
}

# --- SCRIPT SETUP & ARGUMENT PARSING ---

# Smart Project Root Detection
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT=$(pwd)
fi

# Get available actions and experiments
ACTIONS=("launch" "status" "clean" "debug-task")
mapfile -t EXPERIMENT_NAMES < <(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print('\n'.join(EXPERIMENTS.keys()))")

# Interactive or Direct Mode selection
ACTION="$1"
EXPERIMENT_NAME="$2"

if [ -z "$ACTION" ]; then
    echo -e "${C_BLUE}Choose an action to perform:${C_NC}"
    PS3="Enter a number for action: "
    select CHOICE in "${ACTIONS[@]}"; do
        [[ -n "$CHOICE" ]] && { ACTION="$CHOICE"; break; } || echo "Invalid choice."
    done
fi

if [ -z "$EXPERIMENT_NAME" ]; then
    echo -e "${C_BLUE}Choose an experiment:${C_NC}"
    PS3="Enter a number for experiment: "
    select CHOICE in "${EXPERIMENT_NAMES[@]}"; do
        [[ -n "$CHOICE" ]] && { EXPERIMENT_NAME="$CHOICE"; break; } || echo "Invalid choice."
    done
fi
echo

# --- RETRIEVE CONFIG VALUES ---
# This python one-liner can fetch any value, even nested ones, from the config.
get_config_value() {
    python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print(EXPERIMENTS['${EXPERIMENT_NAME}']['${1}'])"
}
get_hpc_param() {
    python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print(EXPERIMENTS['${EXPERIMENT_NAME}']['HPC_PARAMS']['${1}'])"
}

CAMPAIGN_ID=$(get_config_value "CAMPAIGN_ID")
TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/generate_tasks.py"
RESUME_TASK_FILE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}_task_list.txt"

# --- MAIN ACTION DISPATCHER ---
echo -e "${C_GREEN}Action: ${C_YELLOW}${ACTION}${C_GREEN} | Campaign: ${C_YELLOW}${CAMPAIGN_ID}${C_NC}"
echo "-----------------------------------------------------"

case "$ACTION" in
    "launch")
        # --- LAUNCH ACTION ---
        echo "Generating/updating task list..."
        python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}"
        
        if [ ! -s "${RESUME_TASK_FILE}" ]; then
            echo -e "${C_GREEN}Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete.${C_NC}"
            exit 0
        fi

        TOTAL_SIMS_TO_RUN=$(wc -l < "${RESUME_TASK_FILE}")
        SIMS_PER_TASK=$(get_hpc_param "sims_per_task")
        NUM_ARRAY_TASKS=$(( (TOTAL_SIMS_TO_RUN + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
        
        MEM_PER_TASK=$(get_hpc_param "mem")
        TIME_PER_TASK=$(get_hpc_param "time")
        
        echo -e "Total simulations to run: ${C_YELLOW}${TOTAL_SIMS_TO_RUN}${C_NC}"
        echo -e "Simulations per array task: ${C_YELLOW}${SIMS_PER_TASK}${C_NC}"
        echo -e "Number of array tasks needed: ${C_YELLOW}${NUM_ARRAY_TASKS}${C_NC}"
        echo -e "Resources per task: ${C_YELLOW}${MEM_PER_TASK} RAM, ${TIME_PER_TASK} wall time${C_NC}"
        
        confirm_action "Proceed with Slurm submission?"

        LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"
        mkdir -p "${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results" "${LOG_DIR}"

        SUBMIT_OUTPUT=$(sbatch \
            --job-name=${CAMPAIGN_ID} \
            --array=1-${NUM_ARRAY_TASKS}%499 \
            --output=${LOG_DIR}/chunk_task-%A_%a.out \
            --error=${LOG_DIR}/chunk_task-%A_%a.err \
            --cpus-per-task=1 \
            --mem=${MEM_PER_TASK} \
            --time=${TIME_PER_TASK} \
            "${PROJECT_ROOT}/scripts/run_chunk.sh" "${RESUME_TASK_FILE}" "${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results" "${PROJECT_ROOT}" "${SIMS_PER_TASK}")

        JOB_ID=$(echo "${SUBMIT_OUTPUT}" | awk '{print $4}')
        echo -e "\n${C_GREEN}Campaign '${CAMPAIGN_ID}' submitted successfully! Job ID: ${JOB_ID}${C_NC}"
        echo "Monitor with: squeue -u ${USER} -j ${JOB_ID}"
        ;;

    "status")
        # --- STATUS ACTION ---
        echo "Checking status... (this may take a moment to generate task universe)"
        python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}" > /dev/null
        
        TOTAL_TASKS_DEFINED=$(jq -s 'length' ${RESUME_TASK_FILE})
        if [ ! -s "${RESUME_TASK_FILE}" ]; then
            TOTAL_TASKS_DEFINED=0
        fi

        COMPLETED_TASK_COUNT=$(find "${PROJECT_ROOT}/data/${CAMPAIGN_ID}/results" -name "result_*.json" 2>/dev/null | wc -l)
        
        # We need the total universe of tasks, not just the missing ones.
        # So we run the generator without --clean and capture its output.
        UNIVERSE_OUTPUT=$($PROJECT_ROOT/scripts/generate_tasks.py $EXPERIMENT_NAME)
        TOTAL_UNIVERSE=$(echo "$UNIVERSE_OUTPUT" | grep 'Generated a universe of' | awk '{print $5}')

        if [ -z "$TOTAL_UNIVERSE" ] || [ "$TOTAL_UNIVERSE" -eq 0 ]; then
            echo -e "${C_YELLOW}No tasks defined for this campaign.${C_NC}"
            exit 0
        fi
        
        PERCENTAGE=$(awk "BEGIN {if ($TOTAL_UNIVERSE > 0) printf \"%.2f\", $COMPLETED_TASK_COUNT / $TOTAL_UNIVERSE * 100; else print 0}")
        echo -e "\nCampaign Progress: ${C_GREEN}${COMPLETED_TASK_COUNT} / ${TOTAL_UNIVERSE} tasks complete (${PERCENTAGE}%)${C_NC}"
        ;;

    "clean")
        # --- CLEAN ACTION ---
        DATA_DIR="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"
        LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"
        
        echo -e "${C_RED}WARNING: This will permanently delete the following:"
        echo "  - Data:    ${DATA_DIR}"
        echo "  - Logs:    ${LOG_DIR}"
        echo "  - Tasklist: ${RESUME_TASK_FILE}"
        
        confirm_action "Are you absolutely sure you want to delete these files?"
        
        echo "Cleaning up..."
        rm -rf "${DATA_DIR}" "${LOG_DIR}" "${RESUME_TASK_FILE}"
        echo -e "${C_GREEN}Cleanup complete.${C_NC}"
        ;;

    "debug-task")
        # --- DEBUG-TASK ACTION ---
        if [ ! -f "${RESUME_TASK_FILE}" ]; then
            echo "Task file not found. Generating it now..."
            python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}"
        fi
        
        TOTAL_LINES=$(wc -l < "${RESUME_TASK_FILE}")
        read -p "Enter the task line number to debug (1-${TOTAL_LINES}): " LINE_NUM
        
        if ! [[ "$LINE_NUM" =~ ^[0-9]+$ ]] || [ "$LINE_NUM" -lt 1 ] || [ "$LINE_NUM" -gt "$TOTAL_LINES" ]; then
            echo -e "${C_RED}Invalid line number.${C_NC}"
            exit 1
        fi
        
        PARAMS_JSON=$(sed -n "${LINE_NUM}p" "${RESUME_TASK_FILE}")
        
        echo "-----------------------------------------------------"
        echo "Running worker with parameters from line ${LINE_NUM}:"
        echo "${PARAMS_JSON}" | jq # Use jq for pretty-printing if available
        echo "-----------------------------------------------------"
        echo "Worker Output:"
        
        python3 "${PROJECT_ROOT}/src/worker.py" --params "${PARAMS_JSON}" | jq
        ;;

    *)
        echo -e "${C_RED}Error: Unknown action '${ACTION}'${C_NC}"
        exit 1
        ;;
esac