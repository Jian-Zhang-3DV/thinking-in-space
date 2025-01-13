#!/bin/bash
#SBATCH --job-name=eval_llava
#SBATCH --nodes=8
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --partition=gh

# 从命令行接收checkpoint参数
pretrained=${1:-"LLaVA-NeXT/checkpoints/LLaVA-Video-7B-Qwen2"}

# 使用srun启动运行脚本，并传入参数
srun bash eval_multi_node.sh $pretrained