#!/bin/bash
#SBATCH --job-name=eval_llava
#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --partition=gh

pretrained=$1
benchmark=$2
model_base=$3
# 使用srun启动运行脚本，并传入参数
srun bash eval_multi_node.sh $pretrained $benchmark $model_base