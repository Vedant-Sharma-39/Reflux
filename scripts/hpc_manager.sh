#!/bin/bash
# FILE: scripts/hpc_manager.sh (v5.8 - Final, Argument Quoting Fix)
# This version fixes the `sbatch` submission error by properly double-quoting
# the variables passed as arguments, which is a critical shell scripting practice.

set -eo pipefail

# --- Style and Helper Functions ---
readonly C_RESET="\033[0m"; readonly C_RED="\033[1;31m"; readonly C_GREEN="\033[1;32m"; readonly C_BLUE="\033[1;34m"; readonly C_YELLOW="\033[1;33m"; readonly C_CYAN="\033[1;36m"; readonly C_PURPLE="\033[1;35m"
msg() { echo -e "${C_GREEN}> ${1}${C_RESET}"; }; msg_info() { echo -e "${C_CYAN}  [i] ${1}${C_RESET}"; }; msg_warn() { echo -e "${C_YELLOW}  [!] ${1}${C_RESET}"; }; msg_error() { echo -e "${C_RED}  [X] ERROR: ${1}${C_RESET}" >&2; }; headline() { echo -e "\n${C_PURPLE}===== ${1} =====${C_RESET}"; }
confirm_action() { read -p "$(echo -e ${C_YELLOW}"  [?] ${1} [y/N]: "${C_RESET})" -n 1 -r REPLY; echo; if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then msg_error "Operation aborted by user."; exit 1; fi; }

# --- Environment and Prerequisite Checks ---
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module purge >/dev/null 2>&1; module load scipy-stack/2024a
headline "Performing Prerequisite Checks"
if ! command -v python3 &> /dev/null; then msg_error "'python3' command not found."; exit 1; fi

# --- Reliable Path Setup ---
readonly SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
readonly PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")

# --- Python Helper to get Experiment Info ---
get_exp_info() {
    ( cd "${PROJECT_ROOT}" && python3 -c "
import sys; from src.config import EXPERIMENTS
def print_shell_var(k, v): print(f'{k}=\"'+str(v).replace('\"','\\\"')+'\"')
exp_name = sys.argv[1]
if exp_name == '--list-experiments':
    print('\n'.join(EXPERIMENTS.keys())); sys.exit(0)
try:
    e = EXPERIMENTS[exp_name]; h = e.get('hpc_params', {})
    print_shell_var('CAMPAIGN_ID', e.get('campaign_id', 'default'))
    print_shell_var('SIMS_PER_CHUNK', h.get('sims_per_task', 50))
    print_shell_var('MEM_PER_TASK', h.get('mem', '2G'))
    print_shell_var('TIME_PER_TASK', h.get('time', '01:00:00'))
except Exception as err:
    print(f'echo \"ERROR: Could not get config for \'{exp_name}\'. Py err: {err}\" >&2; exit 1;', file=sys.stderr)
    sys.exit(1)
" "$1" )
}

# --- Argument Parsing and Interactive Menus ---
if [[ -z "$1" ]]; then
    headline "HPC Campaign Manager"
    msg "Please choose an action:"
    actions=("Launch Campaign" "Debug a Single Task" "Clean Campaign Data" "Exit")
    PS3="$(echo -e ${C_YELLOW}"  Your choice: "${C_RESET})"
    select choice in "${actions[@]}"; do
        case "$choice" in
            "Launch Campaign") ACTION="launch"; break;;
            "Debug a Single Task") ACTION="debug-task"; break;;
            "Clean Campaign Data") ACTION="clean"; break;;
            "Exit") exit 0;;
            *) msg_warn "Invalid option.";;
        esac
    done

    msg "Please choose an experiment:"
    mapfile -t experiment_names < <(get_exp_info "--list-experiments")
    if [ ${#experiment_names[@]} -eq 0 ]; then msg_error "No experiments in 'src/config.py'."; exit 1; fi
    PS3="$(echo -e ${C_YELLOW}"  Your choice: "${C_RESET})"
    select choice in "${experiment_names[@]}"; do
        if [[ -n "$choice" ]]; then EXPERIMENT_NAME="$choice"; break; else msg_warn "Invalid option."; fi
    done
else
    ACTION="$1"; EXPERIMENT_NAME="$2"
    if [[ -z "$EXPERIMENT_NAME" ]]; then msg_error "Experiment name must be provided in command-line mode."; exit 1; fi
fi

headline "Loading config for '${EXPERIMENT_NAME}'..."
eval "$(get_exp_info "${EXPERIMENT_NAME}")"
if [ -z "${CAMPAIGN_ID}" ]; then msg_error "Failed to parse CAMPAIGN_ID."; exit 1; fi
msg_info "Campaign ID is '${C_YELLOW}${CAMPAIGN_ID}${C_RESET}'."

# --- Define Paths ---
readonly TASK_GEN="scripts/utils/generate_tasks.py"; readonly WORKER="src/worker.py"
readonly DATA_DIR_BASE="${PROJECT_ROOT}/data/${CAMPAIGN_ID}"; readonly RAW_DATA_DIR="${DATA_DIR_BASE}/raw"
readonly MASTER_TASK_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_master_tasks.jsonl"
readonly LOG_DIR="${PROJECT_ROOT}/slurm_logs/${CAMPAIGN_ID}";

# --- Action Execution ---
case "$ACTION" in
    "launch")
        submit_job_array() {
            local job_name="$1"; local task_list_file="$2"; local task_count="$3"
            if [ "$task_count" -eq 0 ]; then msg_info "No tasks to submit."; return; fi
            local num_array_tasks=$(( (task_count + SIMS_PER_CHUNK - 1) / SIMS_PER_CHUNK ))
            msg "Submitting job array '${job_name}'..."
            
            local dummy_job_id=$(sbatch --parsable --job-name=log_alloc -t 1 -o /dev/null --wrap="sleep 1"); scancel "${dummy_job_id}" >/dev/null 2>&1
            local next_job_id=$((dummy_job_id + 1)); local job_log_dir_base="${LOG_DIR}"
            local predicted_job_log_dir="${job_log_dir_base}/${next_job_id}"
            mkdir -p "${predicted_job_log_dir}/running" "${predicted_job_log_dir}/success" "${predicted_job_log_dir}/failed"
            
            # --- [THE FINAL FIX] ---
            # Double-quote the variables passed to sbatch arguments.
            SBATCH_OUTPUT=$(sbatch --job-name="${job_name}" --array=1-"${num_array_tasks}"%500 \
                --output="${predicted_job_log_dir}/running/task_%a.log" \
                --mem="${MEM_PER_TASK}" \
                --time="${TIME_PER_TASK}" \
                "${PROJECT_ROOT}/scripts/run_chunk.sh" \
                "${PROJECT_ROOT}" "${job_log_dir_base}" "${task_list_file}" "${RAW_DATA_DIR}" "${SIMS_PER_CHUNK}" 2>&1)
            # --- [END FIX] ---
            
            local sbatch_exit_code=$?; if [ $sbatch_exit_code -ne 0 ]; then msg_error "sbatch failed:\n${SBATCH_OUTPUT}"; exit 1; fi
            local real_job_id=$(echo "$SBATCH_OUTPUT" | awk '{print $NF}')
            if [ "$real_job_id" != "$next_job_id" ]; then mv "${predicted_job_log_dir}" "${LOG_DIR}/${real_job_id}"; fi
            msg "${C_GREEN}Submitted! Job ID: ${C_YELLOW}${real_job_id}${C_RESET}"; msg_info "Logs will be in: ${C_CYAN}${LOG_DIR}/${real_job_id}${C_RESET}"
        }

        msg "Generating/updating master task list..."; (cd "${PROJECT_ROOT}" && python3 "${TASK_GEN}" "${EXPERIMENT_NAME}")
        if [ ! -s "${MASTER_TASK_FILE}" ]; then msg_warn "Master task file empty."; exit 0; fi
        readonly TOTAL_TASKS=$(wc -l < "${MASTER_TASK_FILE}")
        readonly MISSING_TASKS_FILE="${DATA_DIR_BASE}/${CAMPAIGN_ID}_missing_tasks.jsonl"
        msg "Checking for completed tasks...";
        readonly missing_count=$(python3 -c "
import sys, json, os
master, raw, missing = sys.argv[1:4]
completed = {f.split('.')[0] for f in os.listdir(raw)} if os.path.exists(raw) else set()
count = 0
with open(master, 'r') as fin, open(missing, 'w') as fout:
    for line in fin:
        if json.loads(line).get('task_id') not in completed:
            fout.write(line); count += 1
print(count)
" "${MASTER_TASK_FILE}" "${RAW_DATA_DIR}" "${MISSING_TASKS_FILE}")
        
        readonly COMPLETED_JOBS=$((TOTAL_TASKS - missing_count))
        headline "Campaign Status"
        PERCENTAGE=$(awk "BEGIN {if (\$TOTAL_TASKS > 0) printf \"%.2f\", 100 * ${COMPLETED_JOBS} / ${TOTAL_TASKS}; else print \"0.00\" }")
        echo -e "  - ${C_GREEN}Completed: ${COMPLETED_JOBS}${C_RESET} | ${C_YELLOW}Missing: ${missing_count}${C_RESET} | ${C_BLUE}Total: ${TOTAL_TASKS}${C_RESET} (${PERCENTAGE}%)"

        if [ "$missing_count" -eq 0 ]; then msg "${C_GREEN}All tasks are complete.${C_RESET}"; exit 0; fi
        confirm_action "Submit job array for the ${missing_count} missing tasks?"
        submit_job_array "${CAMPAIGN_ID}" "${MISSING_TASKS_FILE}" "${missing_count}"
        ;;
    "debug-task")
        if [ ! -f "${MASTER_TASK_FILE}" ]; then confirm_action "Master file not found. Generate it?" && (cd "${PROJECT_ROOT}" && python3 "${TASK_GEN}" "${EXPERIMENT_NAME}"); fi
        read -p "Enter line number to debug: " LINE_NUM
        if ! [[ "$LINE_NUM" =~ ^[0-9]+$ ]] || [ "$LINE_NUM" -eq 0 ]; then msg_error "Invalid line number."; exit 1; fi
        PARAMS_JSON=$(sed -n "${LINE_NUM}p" "${MASTER_TASK_FILE}")
        if [ -z "${PARAMS_JSON}" ]; then msg_error "Line ${LINE_NUM} not found."; exit 1; fi
        headline "Debugging Task (Line ${LINE_NUM})"; msg_info "Parameters:"; echo "${PARAMS_JSON}" | python3 -m json.tool
        msg "\n${C_PURPLE}--- Running Worker ---${C_RESET}"
        export PROJECT_ROOT
        (cd "${PROJECT_ROOT}" && python3 "${WORKER}" --params "${PARAMS_JSON}" --output-dir "${RAW_DATA_DIR}")
        TASK_ID=$(echo "${PARAMS_JSON}" | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
        if [ -f "${RAW_DATA_DIR}/${TASK_ID}.json" ]; then msg "\n${C_GREEN}Success${C_RESET}"; msg_info "Output: ${RAW_DATA_DIR}/${TASK_ID}.json";
        elif [ -f "${RAW_DATA_DIR}/${TASK_ID}.error" ]; then msg "\n${C_RED}FAILED${C_RESET}"; msg_info "Error: ${RAW_DATA_DIR}/${TASK_ID}.error"; cat "${RAW_DATA_DIR}/${TASK_ID}.error";
        else msg_warn "No output file created."; fi
        ;;
    "clean")
        msg_warn "This will DELETE all campaign data."; msg_info "Data dir: ${DATA_DIR_BASE}"; msg_info "Log dir:  ${LOG_DIR}"; confirm_action "Are you sure?"; rm -rf "${DATA_DIR_BASE}" "${LOG_DIR}"; msg "${C_GREEN}Cleanup complete.${C_RESET}";
        ;;
    *) msg_error "Unrecognized action '$ACTION'."; exit 1;;
esac