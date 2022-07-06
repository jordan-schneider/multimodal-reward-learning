#!/bin/bash
#SBATCH --job-name=policy-train
#SBATCH --array=1-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=31G
#SBATCH --partition=debug
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jordan.jack.schneider@gmail.com
#SBATCH --output=data/slurm-logs/%A-%a.out
#SBATCH --error=data/slurm-logs/%A-%a.err

./build_env.sh
srun --cpus-per-task=8 conda run -n mrl python mrl/model_training/train_oracles.py --path=data/ --env-name=miner --seed=7892365 --replications=$SLURM_ARRAY_TASK_ID --total-interactions=10_000
