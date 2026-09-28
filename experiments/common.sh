# Shared launcher for the experiment scripts. Source it from a run script; do not execute it.
#
# A run script fills the bash array TASKS with one training command per entry and ends with
# `run_tasks "$@"`. Then, from the repository root,
#
#     bash <script>            runs every task, in order
#     bash <script> 3          runs task 3 only
#     bash <script> list       prints the numbered tasks
#
# and inside a SLURM array job (SLURM_ARRAY_TASK_ID set) a script called without an argument
# runs the task with that index, one task per GPU, e.g.
#
#     sbatch --array=0-11 --gpus=1 --time=04:00:00 --wrap "bash <script>"
#
# PY selects the interpreter (default: python).

PY=${PY:-python}

# lr / 10, formatted by Python exactly as the paper's launchers computed it (3e-4 -> 2.9999999999999997e-05).
lr_div10() { python3 -c "print($1 / 10)"; }

run_tasks() {
    local which="${1:-${SLURM_ARRAY_TASK_ID:-all}}"
    local n=${#TASKS[@]}
    case "$which" in
        list)
            for i in "${!TASKS[@]}"; do printf '[%d] %s\n' "$i" "${TASKS[$i]}"; done ;;
        all)
            for i in "${!TASKS[@]}"; do
                printf '\n>>> task %d/%d: %s\n' "$i" "$n" "${TASKS[$i]}"
                eval "${TASKS[$i]}"
            done ;;
        *)
            if ! [[ "$which" =~ ^[0-9]+$ ]] || (( which >= n )); then
                echo "task must be an index in 0..$((n - 1)), 'list' or 'all' (got '$which')" >&2
                exit 2
            fi
            printf '>>> task %d/%d: %s\n' "$which" "$n" "${TASKS[$which]}"
            eval "${TASKS[$which]}" ;;
    esac
}
