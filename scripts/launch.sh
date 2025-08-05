#!/bin/bash
# FILE: scripts/launch.sh
# [v2 - CONSOLIDATION-AWARE]
# Campaign Management CLI. Automatically consolidates raw data before
# status checks and new job launches.

set -eo pipefail

C_RED="\033[1;31m"; C_GREEN="\033[1;32m"; C_BLUE="\033[1;34m"; C_YELLOW="\033[1;33m"; C_NC="\033[0m"

confirm_action() {
    local prompt="$1"
    read -p "$(echo -e ${C_YELLOW}"${prompt} [y/N]: "${C_NC})" -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then echo -e "${C_RED}Aborted.${C_NC}"; exit 1; fi
}

if [ -n "${SLURM_SUBMIT_DIR}" ]; then PROJECT_ROOT="${SLURM_SUBMIT_DIR}"; else PROJECT_ROOT=$(pwd); fi

# --- ### NEW: Add 'consolidate' to the list of actions ### ---
ACTIONS=("launch" "status" "consolidate" "clean" "debug-task")
mapfile -t EXPERIMENT_NAMES < <(python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print('\n'.join(EXPERIMENTS.keys()))")

ACTION="$1"; EXPERIMENT_NAME="$2"
if [ -z "$ACTION" ]; then
    echo -e "${C_BLUE}Choose an action:${C_NC}"; PS3="Action: "; select CHOICE in "${ACTIONS[@]}"; do [[ -n "$CHOICE" ]] && { ACTION="$CHOICE"; break; }; done
fi
if [ -z "$EXPERIMENT_NAME" ]; then
    echo -e "${C_BLUE}Choose an experiment:${C_NC}"; PS3="Experiment: "; select CHOICE in "${EXPERIMENT_NAMES[@]}"; do [[ -n "$CHOICE" ]] && { EXPERIMENT_NAME="$CHOICE"; break; }; done
fi; echo

get_config_value() { python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print(EXPERIMENTS['${EXPERIMENT_NAME}']['${1}'])"; }
get_hpc_param() { python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from config import EXPERIMENTS; print(EXPERIMENTS['${EXPERIMENT_NAME}']['HPC_PARAMS']['${1}'])"; }

CAMPAIGN_ID=$(get_config_value "CAMPAIGN_ID")
TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/generate_tasks.py"
DATA_DIR_BASE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"
RESUME_TASK_FILE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}_task_list.txt"

echo -e "${C_GREEN}Action: ${C_YELLOW}${ACTION}${C_GREEN} | Campaign: ${C_YELLOW}${CAMPAIGN_ID}${C_NC}"
echo "-----------------------------------------------------"

case "$ACTION" in
    "launch"|"status")
        # --- ### NEW: Auto-consolidation ### ---
        echo "Checking for and consolidating raw data..."
        python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from data_utils import consolidate_raw_data; consolidate_raw_data('${CAMPAIGN_ID}', '${PROJECT_ROOT}')"

        echo "Generating/updating task list..."
        python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}" > /dev/null

        MASTER_SUMMARY_FILE="${DATA_DIR_BASE}/analysis/${CAMPAIGN_ID}_summary_aggregated.csv"
        COMPLETED_TASK_COUNT=0
        if [ -f "${MASTER_SUMMARY_FILE}" ]; then
            # Count lines in master CSV, subtract 1 for header
            COMPLETED_TASK_COUNT=$(($(wc -l < "${MASTER_SUMMARY_FILE}") - 1))
        fi
        
        UNIVERSE_OUTPUT=$($PROJECT_ROOT/scripts/generate_tasks.py $EXPERIMENT_NAME)
        TOTAL_UNIVERSE=$(echo "$UNIVERSE_OUTPUT" | grep 'Generated a universe of' | awk '{print $5}')

        if [ "$ACTION" == "status" ]; then
            if [ -z "$TOTAL_UNIVERSE" ] || [ "$TOTAL_UNIVERSE" -eq 0 ]; then
                echo -e "${C_YELLOW}No tasks defined for this campaign.${C_NC}"; exit 0
            fi
            PERCENTAGE=$(awk "BEGIN {if ($TOTAL_UNIVERSE > 0) printf \"%.2f\", $COMPLETED_TASK_COUNT / $TOTAL_UNIVERSE * 100; else print 0}")
            echo -e "\nCampaign Progress: ${C_GREEN}${COMPLETED_TASK_COUNT} / ${TOTAL_UNIVERSE} tasks complete (${PERCENTAGE}%)${C_NC}"
            exit 0
        fi
        
        # --- Launch logic (mostly unchanged) ---
        if [ ! -s "${RESUME_TASK_FILE}" ]; then
            echo -e "${C_GREEN}Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete.${C_NC}"; exit 0
        fi
        TOTAL_SIMS_TO_RUN=$(wc -l < "${RESUME_TASK_FILE}")
        SIMS_PER_TASK=$(get_hpc_param "sims_per_task")
        NUM_ARRAY_TASKS=$(( (TOTAL_SIMS_TO_RUN + SIMS_PER_TASK - 1) / SIMS_PER_TASK ))
        MEM_PER_TASK=$(get_hpc_param "mem"); TIME_PER_TASK=$(get_hpc_param "time")
        
        echo -e "Total simulations to run: ${C_YELLOW}${TOTAL_SIMS_TO_RUN}${C_NC}"; confirm_action "Proceed?"
        
        LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"; mkdir -p "${LOG_DIR}"
        # --- ### NEW: Use single, combined log file ### ---
        sbatch \
            --job-name=${CAMPAIGN_ID} \
            --array=1-${NUM_ARRAY_TASKS}%499 \
            --output=${LOG_DIR}/chunk_task-%A_%a.log \
            --cpus-per-task=1 --mem=${MEM_PER_TASK} --time=${TIME_PER_TASK} \
            "${PROJECT_ROOT}/scripts/run_chunk.sh" "${RESUME_TASK_FILE}" "${DATA_DIR_BASE}" "${PROJECT_ROOT}" "${SIMS_PER_TASK}"

        echo -e "\n${C_GREEN}Campaign '${CAMPAIGN_ID}' submitted successfully!${C_NC}"
        ;;

    "consolidate")
        echo "Manually consolidating raw data..."
        python3 -c "import sys; sys.path.insert(0, '${PROJECT_ROOT}/src'); from data_utils import consolidate_raw_data; consolidate_raw_data('${CAMPAIGN_ID}', '${PROJECT_ROOT}')"
        echo "Consolidation complete."
        ;;
    
    # ... (clean and debug-task actions are unchanged) ...
esac