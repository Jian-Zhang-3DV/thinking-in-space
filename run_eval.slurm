#!/bin/bash
#SBATCH --job-name=eval_llava
#SBATCH --nodes=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --partition=gh

# 从命令行接收checkpoint参数
pretrained=$1
model_name=$2
model_base=$3
# 使用srun启动运行脚本，并传入参数
srun bash eval_multi_node.sh $pretrained $model_name $model_base