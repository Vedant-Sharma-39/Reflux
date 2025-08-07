#!/bin/bash
# FILE: scripts/hpc_manager.sh (v5.0 - Production Ready)
# This is the final, complete version. It uses the robust 'eval' method for
# configuration and removes the invalid --partition flag to allow the scheduler
# to auto-assign the job, fixing the submission failure.

set -eo pipefail

# --- Style and Helper Functions ---
readonly C_RESET="\033[0m"; readonly C_RED="\033[1;31m"; readonly C_GREEN="\033[1;32m"; readonly C_BLUE="\033[1;34m"; readonly C_YELLOW="\033[1;33m"; readonly C_CYAN="\033[1;36m"; readonly C_PURPLE="\033[1;35m"
msg() { echo -e "${C_GREEN}> ${1}${C_RESET}"; }; msg_info() { echo -e "${C_CYAN}  [i] ${1}${C_RESET}"; }; msg_warn() { echo -e "${C_YELLOW}  [!] ${1}${C_RESET}"; }; msg_error() { echo -e "${C_RED}  [X] ERROR: ${1}${C_RESET}" >&2; }; headline() { echo -e "\n${C_PURPLE}===== ${1} =====${C_RESET}"; }
confirm_action() { read -p "$(echo -e ${C_YELLOW}"  [?] ${1} [y/N]: "${C_RESET})" -n 1 -r REPLY; echo; if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then msg_error "Operation aborted by user."; exit 1; fi; }

# ==============================================================================
# 1. ENVIRONMENT SETUP
# ==============================================================================
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module purge >/dev/null 2>&1
module load scipy-stack/2024a

# ==============================================================================
# 2. PREREQUISITE CHECKS
# ==============================================================================
headline "Performing Prerequisite Checks"
if ! command -v python3 &> /dev/null; then msg_error "The 'python3' command was not found."; exit 1; fi
msg_info "Python 3 interpreter found at: $(command -v python3)"
if ! python3 -c "import pandas" &> /dev/null; then msg_error "Python environment is missing 'pandas'."; exit 1; fi
msg_info "Key Python libraries are available."

# --- Reliable Path Setup ---
readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

# --- Python Helper to get Experiment Info (eval-safe version) ---
get_exp_info() {
    (
        cd "${PROJECT_ROOT}" && \
        python3 -c "
import sys
def print_shell_var(key, value):
    sanitized_value = str(value).replace('\"', '\\\"')
    print(f'{key}=\"{sanitized_value}\"')
from src.config import EXPERIMENTS
exp_name = sys.argv[1]
if exp_name == '--list-experiments':
    if not EXPERIMENTS: print('ERROR: EXPERIMENTS dict is empty.', file=sys.stderr); sys.exit(1)
    print('\n'.join(EXPERIMENTS.keys()))
    sys.exit(0)
try:
    e = EXPERIMENTS[exp_name]
    h = e.get('hpc_params', {})
    print_shell_var('CAMPAIGN_ID', e.get('campaign_id', 'default_campaign'))
    print_shell_var('SIMS_PER_CHUNK', h.get('sims_per_task', 50))
    print_shell_var('MEM_PER_TASK', h.get('mem', '2G'))
    print_shell_var('TIME_PER_TASK', h.get('time', '01:00:00'))
except Exception as err:
    print(f'echo \"ERROR: Could not get config for experiment \'{exp_name}\'. Python error: {err}\" >&2; exit 1;', file=sys.stderr)
    sys.exit(1)
" "$1"
    )
}

# --- Interactive Menu Functions ---
select_action() { headline "HPC Campaign Manager"; msg "Please choose an action."; local actions=("Launch Campaign" "Debug a Single Task" "Clean Campaign Data" "Exit"); PS3="$(echo -e ${C_YELLOW}"  Your choice: "${C_RESET})"; select choice in "${actions[@]}"; do case "$choice" in "Launch Campaign") ACTION="launch"; break ;; "Debug a Single Task") ACTION="debug-task"; break ;; "Clean Campaign Data") ACTION="clean"; break ;; "Exit") exit 0 ;; *) msg_warn "Invalid option." ;; esac; done; }
select_experiment() { msg "Please choose an experiment."; mapfile -t experiment_names < <(get_exp_info "--list-experiments"); if [ ${#experiment_names[@]} -eq 0 ]; then msg_error "No experiments in 'src/config.py'."; exit 1; fi; PS3="$(echo -e ${C_YELLOW}"  Your choice: "${C_RESET})"; select choice in "${experiment_names[@]}"; do if [[ -n "$choice" ]]; then EXPERIMENT_NAME="$choice"; break; else msg_warn "Invalid option."; fi; done; }

# --- Main Logic ---
select_action
select_experiment

headline "Loading configuration for '${EXPERIMENT_NAME}'..."
eval "$(get_exp_info "${EXPERIMENT_NAME}")"
if [ -z "${CAMPAIGN_ID}" ]; then msg_error "Failed to parse configuration: CAMPAIGN_ID is empty."; exit 1; fi
msg_info "Configuration loaded successfully. Campaign ID is '${C_YELLOW}${CAMPAIGN_ID}${C_RESET}'."

# --- Define Paths and other scripts---
readonly TASK_GENERATOR_SCRIPT="scripts/utils/generate_tasks.py"
readonly CONSOLIDATOR_SCRIPT="scripts/utils/consolidate_data.py"
readonly EXTRACTOR_SCRIPT="scripts/utils/extract_failed_params.py"
readonly WORKER_SCRIPT="src/worker.py"
readonly DATA_DIR_BASE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"
readonly RAW_DATA_DIR="${DATA_DIR_BASE}/raw"
readonly MASTER_TASK_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_master_tasks.jsonl"
readonly LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}";

# --- Action Execution ---
case "$ACTION" in
    "launch")
        if [ ! -f "${MASTER_TASK_FILE}" ]; then
            msg_warn "Master task file not found."; msg_info "Path: ${C_CYAN}${MASTER_TASK_FILE}${C_RESET}"
            confirm_action "Generate now?"
            (cd "${PROJECT_ROOT}" && python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}")
            msg "${C_GREEN}Generated $(wc -l < "${MASTER_TASK_FILE}") tasks.${C_RESET}"
        fi
        readonly TOTAL_TASKS=$(wc -l < "${MASTER_TASK_FILE}")
        if [ ! -d "${RAW_DATA_DIR}" ]; then mkdir -p "${RAW_DATA_DIR}"; fi
        readonly COMPLETED_JOBS=$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name "*.json" 2>/dev/null | wc -l)
        readonly FAILED_JOBS=$(find "${RAW_DATA_DIR}" -maxdepth 1 -type f -name "*.error" 2>/dev/null | wc -l)
        headline "Campaign Status"; if [ "${TOTAL_TASKS}" -eq 0 ]; then msg_warn "Master file empty."; exit 0; fi
        PERCENTAGE=$(awk "BEGIN {if (\$TOTAL_TASKS > 0) printf \"%.2f\", \$COMPLETED_JOBS / \$TOTAL_TASKS * 100; else print \"0.00\"}")
        echo -e "  - ${C_GREEN}Completed: ${COMPLETED_JOBS}${C_RESET} | ${C_RED}Failed: ${FAILED_JOBS}${C_RESET} | ${C_BLUE}Total: ${TOTAL_TASKS}${C_RESET} (${PERCENTAGE}%)"
        
submit_job_array() {
            local job_name="$1"; local task_list_file="$2"; local task_count="$3"
            local num_array_tasks=$(( (task_count + SIMS_PER_CHUNK - 1) / SIMS_PER_CHUNK ))
            if [ "$num_array_tasks" -eq 0 ]; then msg_info "No tasks to submit."; return; fi
            
            msg "Submitting job array '${job_name}'..."
            
            # This is a placeholder command to get the next job ID for directory creation.
            # It submits a dummy job that sleeps for 1 second and then immediately cancels it.
            # This is a robust way to reserve a Job ID for our log directory.
            local dummy_job_id=$(sbatch --parsable --job-name=log_alloc -t 1 -o /dev/null --wrap="sleep 1")
            scancel "${dummy_job_id}" >/dev/null 2>&1
            local next_job_id=$((dummy_job_id + 1))
            
            # --- [IMPROVEMENT] Create a structured, job-specific log directory ---
            local job_log_dir="${LOG_DIR}/${next_job_id}"
            mkdir -p "${job_log_dir}/running"
            mkdir -p "${job_log_dir}/success"
            mkdir -p "${job_log_dir}/failed"
            msg_info "Logs for this submission will be in: ${C_CYAN}${job_log_dir}${C_RESET}"

            # The --partition flag has been removed from this call.
            # We now pass the main Job ID (%A) to the script for log handling.
            SBATCH_OUTPUT=$(sbatch --job-name="${job_name}" --array=1-"${num_array_tasks}"%500 \
                --output="${job_log_dir}/running/task_%a.log" \
                --mem="${MEM_PER_TASK}" --time="${TIME_PER_TASK}" \
                "${PROJECT_ROOT}/scripts/run_chunk.sh" \
                "${job_log_dir}" \
                "${task_list_file}" \
                "${RAW_DATA_DIR}" \
                "${SIMS_PER_CHUNK}" 2>&1)
            
            SBATCH_EXIT_CODE=$?
            if [ $SBATCH_EXIT_CODE -ne 0 ]; then msg_error "sbatch command failed! Slurm error:\n${C_RED}${SBATCH_OUTPUT}${C_RESET}"; exit 1; fi
            
            local real_job_id=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
            if [ "$real_job_id" != "$next_job_id" ]; then
                msg_warn "Log directory name and actual Job ID mismatch (${next_job_id} vs ${real_job_id}). Renaming log directory."
                mv "${job_log_dir}" "${LOG_DIR}/${real_job_id}"
            fi

            msg "${C_GREEN}Submitted! Job ID: ${C_YELLOW}${real_job_id}${C_RESET}"
        }
        
        if [ "${COMPLETED_JOBS}" -eq "${TOTAL_TASKS}" ]; then
            msg "${C_GREEN}All tasks complete.${C_RESET}"; confirm_action "Consolidate data?"
            (cd "${PROJECT_ROOT}" && python3 "${CONSOLIDATOR_SCRIPT}" "${CAMPAIGN_ID}")
            msg "${C_GREEN}Consolidation complete.${C_RESET}";
        elif [ "${FAILED_JOBS}" -gt 0 ]; then
            msg_warn "Found failed jobs."; confirm_action "Resubmit ${FAILED_JOBS} failed tasks?"
            readonly FAILED_LIST_FILE=$(mktemp); trap 'rm -f -- "$FAILED_LIST_FILE"' EXIT
            (cd "${PROJECT_ROOT}" && find "${RAW_DATA_DIR}" -name "*.error" -print0 | xargs -0 python3 "${EXTRACTOR_SCRIPT}" > "${FAILED_LIST_FILE}")
            find "${RAW_DATA_DIR}" -name "*.error" -delete
            submit_job_array "${CAMPAIGN_ID}_retry" "${FAILED_LIST_FILE}" "${FAILED_JOBS}";
        else
            msg "Campaign ready to run."; confirm_action "Submit job array for all ${TOTAL_TASKS} tasks?"
            submit_job_array "${CAMPAIGN_ID}" "${MASTER_TASK_FILE}" "${TOTAL_TASKS}";
        fi
        ;;
    "debug-task")
        if [ ! -f "${MASTER_TASK_FILE}" ]; then
            msg_warn "Master task file not found for this experiment."
            confirm_action "Generate it now to proceed with debugging?"
            (cd "${PROJECT_ROOT}" && python3 "${TASK_GENERATOR_SCRIPT}" "${EXPERIMENT_NAME}")
            msg "${C_GREEN}Generated $(wc -l < "${MASTER_TASK_FILE}") tasks.${C_RESET}"
        fi
        msg "Enter line number to debug."; read -p "$(echo -e ${C_YELLOW}"  Line number: "${C_RESET})" LINE_NUM
        if ! [[ "$LINE_NUM" =~ ^[0-9]+$ ]] || [ "$LINE_NUM" -eq 0 ]; then msg_error "Invalid line number."; exit 1; fi
        PARAMS_JSON=$(sed -n "${LINE_NUM}p" "${MASTER_TASK_FILE}")
        if [ -z "${PARAMS_JSON}" ]; then msg_error "Line ${LINE_NUM} not found."; exit 1; fi
        headline "Debugging Task (Line ${LINE_NUM})"; msg_info "Parameters:"; echo "${PARAMS_JSON}" | python3 -m json.tool
        msg "\n${C_PURPLE}--- Running Python Worker Interactively ---${C_RESET}"
        (cd "${PROJECT_ROOT}" && python3 "${WORKER_SCRIPT}" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}")
        TASK_ID=$(echo "${PARAMS_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
        if [ -f "${RAW_DATA_DIR}/${TASK_ID}.json" ]; then msg "\n${C_GREEN}Worker Finished Successfully${C_RESET}"; msg_info "Output: ${C_CYAN}${RAW_DATA_DIR}/${TASK_ID}.json${C_RESET}";
        elif [ -f "${RAW_DATA_DIR}/${TASK_ID}.error" ]; then msg "\n${C_RED}Worker FAILED${C_RESET}"; msg_info "Error: ${C_CYAN}${RAW_DATA_DIR}/${TASK_ID}.error${C_RESET}"; msg_info "Content:"; cat "${RAW_DATA_DIR}/${TASK_ID}.error";
        else msg_warn "Worker finished, but no output file was created."; fi
        ;;
    "clean")
        msg_warn "This will PERMANENTLY DELETE all campaign data."; msg_info "Data dir: ${C_CYAN}${DATA_DIR_BASE}${C_RESET}"; msg_info "Log dir:  ${C_CYAN}${LOG_DIR}${C_RESET}"; confirm_action "Are you sure?"; rm -rf "${DATA_DIR_BASE}" "${LOG_DIR}"; msg "${C_GREEN}Cleanup complete.${C_RESET}";
        ;;
    *) msg_error "Internal error: Unrecognized action '$ACTION'."; exit 1;;
esac