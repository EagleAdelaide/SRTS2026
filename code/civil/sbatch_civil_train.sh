#!/bin/bash
#SBATCH --job-name=www27_civil
#SBATCH --partition=a100,singlegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --array=0-4
#SBATCH --output=__SRTS_ROOT__/calattn-repo/logs/www2027/civil_train_%A_%a.out
# CivilComments validation backbone (amended protocol, frozen 2026-10-22).
# GPU cap: 5 tasks x 1 GPU x 1h15m time limit = 6.25 A100-hours maximum (author cap 8, aggregate).
set -e
export HF_HOME=__SRTS_ROOT__/hf_home
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
source __SRTS_ROOT__/envs/www27/bin/activate
SEED=${SLURM_ARRAY_TASK_ID}
echo "GPUSTAMP start $(date +%s) job ${SLURM_ARRAY_JOB_ID}_${SEED} host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python __SRTS_ROOT__/calattn-repo/www2027/scripts/civil/train_civil.py --seed ${SEED}
echo "GPUSTAMP end $(date +%s) job ${SLURM_ARRAY_JOB_ID}_${SEED}"
