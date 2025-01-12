#!/bin/bash
#SBATCH --job-name=eval_llava
#SBATCH --nodes=8
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --partition=gh

# 使用srun启动运行脚本
srun bash eval_multi_node.sh