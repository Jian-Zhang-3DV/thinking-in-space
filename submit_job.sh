#!/bin/bash -l


#SBATCH --gres=gpu:a100:2       # 请求4个 A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G               # 请求64 GB内存
#SBATCH -t 5:00:00              # 作业运行时间为 5 小时
#SBATCH --job-name=eval-job     # 作业名称
#SBATCH -p gpu                  # 指定分区

# 运行 eval.sh 脚本
./eval.sh
