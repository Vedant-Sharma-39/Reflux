#!/bin/bash
# FILE: scripts/launch.sh
# [v_FINAL_HYBRID_COMPLETE] Manages the simulation lifecycle using a master task list
# and a "smart chunking" approach to respect HPC array limits.
# This version includes interactive menus and full informational summaries.

set -eo pipefail

# --- Style and Helper Functions ---
readonly C_RED="\033[1;31m"; readonly C_GREEN="\033[1;32m"; readonly C_BLUE="\033[1;34m"; readonly C_YELLOW="\033[1;33m"; readonly C_NC="\033[0m"

confirm_action() {
    read -p "$(echo -e ${C_YELLOW}"$1 [y/N]: "${C_NC})" -n 1 -r; echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then echo -e "${C_RED}Aborted by user.${C_NC}"; exit 1; fi
}

# Set PROJECT_ROOT once, reliably.
if [ -n "${SLURM_SUBMIT_DIR}" ]; then readonly PROJECT_ROOT="${SLURM_SUBMIT_DIR}"; else readonly PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || realpath "$(dirname "$0")/../.."); fi
export PROJECT_ROOT # Export for worker script to use

# --- Python Helper to get Experiment Info ---
# This single function handles listing experiments and fetching config for one.
get_exp_info() {
    python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}'); from src.config_loader import get_experiment_config, EXPERIMENTS
if '$1' == '--list-experiments':
    print('\n'.join(EXPERIMENTS.keys()))
    sys.exit(0)
try:
    e = get_experiment_config('$1')
    h = e.get('hpc_params', {})
    print(e.get('campaign_id', 'default_campaign'))
    print(h.get('sims_per_task', 50)) # Note: This is now sims_per_chunk
    print(h.get('mem', '2G'))
    print(h.get('time', '01:00:00'))
except Exception as err:
    print(f'ERROR: {err}', file=sys.stderr); sys.exit(1)
"
}

# --- Argument Parsing and UI ---
ACTION="${1:-}"; EXPERIMENT_NAME="${2:-}"

# Interactive Action Menu
if [ -z "$ACTION" ]; then
    readonly ACTIONS=("launch" "status" "generate" "consolidate" "clean" "debug-task")
    echo -e "${C_BLUE}Choose an action:${C_NC}"
    PS3="Action: "
    select CHOICE in "${ACTIONS[@]}"; do
        [[ -n "$CHOICE" ]] && { ACTION="$CHOICE"; break; }
    done
fi

# Interactive Experiment Menu (runs if needed)
if [[ " launch status generate consolidate clean debug-task " =~ " ${ACTION} " ]] && [ -z "$EXPERIMENT_NAME" ]; then
    mapfile -t EXPERIMENT_NAMES < <(get_exp_info "--list-experiments")
    if [ ${#EXPERIMENT_NAMES[@]} -eq 0 ]; then
        echo -e "${C_RED}Could not load any experiments. Check config.yml and config_loader.py${C_NC}"; exit 1;
    fi
    echo -e "${C_BLUE}Choose an experiment:${C_NC}"
    PS3="Experiment: "
    select CHOICE in "${EXPERIMENT_NAMES[@]}"; do
        [[ -n "$CHOICE" ]] && { EXPERIMENT_NAME="$CHOICE"; break; }
    done
fi; echo

# --- Load Configuration ---
if [ -n "$EXPERIMENT_NAME" ]; then
    IFS=$'\n' read -r -d '' CAMPAIGN_ID SIMS_PER_CHUNK MEM_PER_TASK TIME_PER_TASK < <(get_exp_info "${EXPERIMENT_NAME}" || exit 1)
fi

# --- Define Paths ---
readonly TASK_GENERATOR_SCRIPT="${PROJECT_ROOT}/scripts/utils/generate_tasks.py"
readonly CONSOLIDATOR_SCRIPT="${PROJECT_ROOT}/scripts/utils/consolidate_data.py"
readonly DATA_DIR_BASE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"
readonly RAW_DATA_DIR="${DATA_DIR_BASE}/raw"
readonly MASTER_TASK_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_master_tasks.jsonl"
readonly RUN_LIST_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_run_list.jsonl"

echo -e "${C_GREEN}Action: ${C_YELLOW}${ACTION}${C_GREEN} | Campaign: ${C_YELLOW}${CAMPAIGN_ID}${C_NC}"
echo "-----------------------------------------------------"

# --- Main Action Logic ---
case "$ACTION" in
    "generate")
        echo "Generating master task list for '${EXPERIMENT_NAME}'..."
        python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}"
        ;;

    "launch"|"status")
        if [ ! -f "${MASTER_TASK_FILE}" ]; then
            echo -e "${C_RED}Error: Master task file not found.${C_NC}"
            echo "Run './scripts/launch.sh generate ${EXPERIMENT_NAME}' first."
            exit 1
        fi
        echo "Scanning for missing tasks..."
        mkdir -p "${RAW_DATA_DIR}"
        TOTAL_TASKS=0
        MISSING_TASKS=0
        > "${RUN_LIST_FILE}" # Create/clear the dynamic run list
        
        while IFS= read -r task_json; do
            ((TOTAL_TASKS++))
            # Efficiently extract task_id using grep for speed
            task_id=$(echo "${task_json}" | grep -o '"task_id":\s*"[^"]*"' | grep -o '[0-9]\+')
            if [ ! -f "${RAW_DATA_DIR}/${task_id}.json" ] && [ ! -f "${RAW_DATA_DIR}/${task_id}.error" ]; then
                ((MISSING_TASKS++))
                echo "${task_json}" >> "${RUN_LIST_FILE}"
            fi
        done < "${MASTER_TASK_FILE}"

        COMPLETED_TASKS=$((TOTAL_TASKS - MISSING_TASKS))
        PERCENTAGE=$(awk "BEGIN {if ($TOTAL_TASKS > 0) printf \"%.2f\", $COMPLETED_TASKS / $TOTAL_TASKS * 100; else print \"100.00\"}")
        
        echo -e "\nCampaign Progress: ${C_GREEN}${COMPLETED_TASKS} / ${TOTAL_TASKS} tasks complete (${PERCENTAGE}%)${C_NC}"

        if [ "$ACTION" == "status" ]; then rm -f "${RUN_LIST_FILE}"; exit 0; fi

        if [ ${MISSING_TASKS} -eq 0 ]; then
            echo -e "${C_GREEN}Congratulations! All tasks for campaign '${CAMPAIGN_ID}' are complete.${C_NC}"
            rm -f "${RUN_LIST_FILE}"
            exit 0
        fi

        readonly NUM_ARRAY_TASKS=$(( (MISSING_TASKS + SIMS_PER_CHUNK - 1) / SIMS_PER_CHUNK ))
        
        echo -e "\nAbout to submit ${C_YELLOW}${NUM_ARRAY_TASKS}${C_NC} array jobs for the ${C_YELLOW}${MISSING_TASKS}${C_NC} remaining tasks."
        echo -e "HPC Config: ${C_BLUE}${SIMS_PER_CHUNK}${C_NC} sims/chunk, ${C_BLUE}${MEM_PER_TASK}${C_NC} memory, ${C_BLUE}${TIME_PER_TASK}${C_NC} time."
        
        confirm_action "Proceed with submission?"
        
        readonly LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"; mkdir -p "${LOG_DIR}"
        
        sbatch --job-name="${CAMPAIGN_ID}" --array=1-"${NUM_ARRAY_TASKS}"%500 --output="${LOG_DIR}/chunk-%A_%a.log" --cpus-per-task=1 --mem="${MEM_PER_TASK}" --time="${TIME_PER_TASK}" \
            "${PROJECT_ROOT}/scripts/run_chunk.sh" "${RUN_LIST_FILE}" "${RAW_DATA_DIR}" "${SIMS_PER_CHUNK}"
        
        echo -e "\n${C_GREEN}Campaign '${CAMPAIGN_ID}' submitted successfully!${C_NC}"
        ;;

    "consolidate")
        echo "Consolidating raw data for ${EXPERIMENT_NAME}..."
        python3 "${CONSOLIDATOR_SCRIPT}" "${CAMPAIGN_ID}"
        ;;

    "clean")
        if [ -z "$CAMPAIGN_ID" ]; then echo -e "${C_RED}Cannot clean without an experiment name.${C_NC}"; exit 1; fi
        confirm_action "Are you sure you want to DELETE ALL data for ${CAMPAIGN_ID}?"
        rm -rf "${DATA_DIR_BASE}" "${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}"
        echo "Cleanup complete."
        ;;

    "debug-task")
        if [ ! -f "${MASTER_TASK_FILE}" ]; then echo "Master task file not found. Run 'generate' first."; exit 1; fi
        read -p "Enter line number from master task file to debug: " LINE_NUM
        PARAMS_JSON=$(sed -n "${LINE_NUM}p" "${MASTER_TASK_FILE}")
        if [ -z "${PARAMS_JSON}" ]; then echo "${C_RED}Error: Line ${LINE_NUM} not found in master file.${C_NC}"; exit 1; fi

        echo "--- Running worker with parameters: ---"; echo "${PARAMS_JSON}" | python3 -m json.tool
        echo "--- Worker Output: ---"
        python3 "${PROJECT_ROOT}/src/worker.py" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}"
        ;;

    *)
        echo -e "${C_RED}Invalid action: $ACTION. Valid actions are: launch, status, generate, consolidate, clean, debug-task.${C_NC}" >&2
        exit 1
        ;;
esac