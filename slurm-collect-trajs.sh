#!/bin/bash
#SBATCH --job-name=policy-train
#SBATCH --array=1-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=29G
#SBATCH --partition=debug
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jordan.jack.schneider@gmail.com
#SBATCH --output=data/slurm-logs/%A-%a.out
#SBATCH --error=data/slurm-logs/%A-%a.err

POLICY_DIR=data/miner/$SLURM_ARRAY_TASK_ID/models
LAST_MODEL=$(ls $POLICY_DIR | grep -o -E "[0-9]+" | sort -n | tail -n 1)

# n-envs Determined empirically

./build_env.sh
srun --cpus-per-task=8 --immediate conda run -n mrl time python mrl/dataset/collect_trajs.py \
  --policies \
    $POLICY_DIR/model$LAST_MODEL.jd \
    $POLICY_DIR/model500.jd \
    $POLICY_DIR/model1000.jd \
    $POLICY_DIR/model1500.jd \
  --outdir data/miner/$SLURM_ARRAY_TASK_ID/ \
  --timesteps 2000 \
  --n-envs 1024 \
  --seed 1978345789
