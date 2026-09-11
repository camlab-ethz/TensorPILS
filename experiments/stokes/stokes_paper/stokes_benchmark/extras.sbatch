#!/bin/bash
# Stokes Table-1 benchmark -- EXTRAS, submitted to Euler's rtx_4090 queue (one GPU per task):
#   0-2  PLS (block P) with the Schur weight tuned like the baselines' continuity weight:
#        omega in {4, 16} at lr 1e-3 / 3e-3 (omega=4, lr 1e-3 runs on the compute node) -- 65^2
#   3-5  PLS with the MONOLITHIC preconditioner (P ~ K^-1, applied_fe form, kappa = O(1)),
#        lr in {3e-4, 1e-3, 3e-3} -- 65^2, ~15x the per-step cost of the block P
#   6-9  mesh refinement to 129^2 (pressure 65^2): PLS block omega 0.5 and 16, PINO at its
#        selected (lr 3e-3, w 300), supervised -- the h-dependence the theory is about
# All at the table's protocol otherwise (K=10, 1024/128/256, 500 epochs, batch 32, Adam, cosine
# lr -> lr/10, seed 42). Outputs go to diagnostics/ (not part of the automatic selection).
#
#   sbatch experiments/stokes/stokes_paper/stokes_benchmark/extras.sbatch
#
#SBATCH --job-name=pils-stokes-extras
#SBATCH --account=ls_math
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=logs/stokes_extras_%A_%a.out
#SBATCH --array=0-9

set -eo pipefail
cd /cluster/work/math/shiwen/projects/tensorpils/code/TensorPILS
module load stack/.2024-04-silent gcc/8.5.0 mesa-glu/9.0.2 ffmpeg 2>/dev/null || true
source ~/venvs/tensorgalerkin/bin/activate
export OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True

COMMON="--pde stokes -k 10 --n_train 1024 --n_val 128 --n_test 256 --epochs 500 --batch_size 32 --optimizer adam --device cuda --seed 42"
MONO="--model fno --loss pls --stokes_precond monolithic --schur_omega 1.0 --uzawa_pre 4 --uzawa_post 4 --cheb_degree 8"
CELLS=(
  "diagnostics/pls_omega/lr1e-3_w16          --grid_resolution 65  --model fno --loss pls --stokes_precond block --schur_omega 16  --lr 1e-3"
  "diagnostics/pls_omega/lr3e-3_w4           --grid_resolution 65  --model fno --loss pls --stokes_precond block --schur_omega 4   --lr 3e-3"
  "diagnostics/pls_omega/lr3e-3_w16          --grid_resolution 65  --model fno --loss pls --stokes_precond block --schur_omega 16  --lr 3e-3"
  "diagnostics/monolithic/lr3e-4             --grid_resolution 65  ${MONO} --lr 3e-4"
  "diagnostics/monolithic/lr1e-3             --grid_resolution 65  ${MONO} --lr 1e-3"
  "diagnostics/monolithic/lr3e-3             --grid_resolution 65  ${MONO} --lr 3e-3"
  "diagnostics/gr129/pls_w0.5_lr1e-3         --grid_resolution 129 --stokes_ref_chunk 32 --model fno --loss pls --stokes_precond block --schur_omega 0.5 --lr 1e-3"
  "diagnostics/gr129/pls_w16_lr1e-3          --grid_resolution 129 --stokes_ref_chunk 32 --model fno --loss pls --stokes_precond block --schur_omega 16  --lr 1e-3"
  "diagnostics/gr129/pino_dw300_lr3e-3       --grid_resolution 129 --stokes_ref_chunk 32 --model fno --loss pino --pi_div_weight 300 --lr 3e-3"
  "diagnostics/gr129/data_lr3e-4             --grid_resolution 129 --stokes_ref_chunk 32 --model fno --loss data --lr 3e-4"
)
read -r SUB FLAGS <<< "${CELLS[$SLURM_ARRAY_TASK_ID]}"
LR=$(echo "$FLAGS" | grep -oE -- '--lr [0-9.e-]+' | awk '{print $2}')
LR_MIN=$(python3 -c "print(${LR}/10)")
OUT=output/stokes/stokes_paper/stokes_benchmark/${SUB}
mkdir -p "$OUT" logs
echo "=== stokes extras task ${SLURM_ARRAY_TASK_ID}: ${SUB} on $(hostname) ($(date)) ==="
python -m tensorpils.cli ${COMMON} ${FLAGS} --lr_min "${LR_MIN}" --output_dir "${OUT}" 2>&1 | tee "${OUT}/train.log" | grep -vE 'it/s\]|s/epoch\]' 
echo "=== done $(date) ==="
