#!/bin/bash
#SBATCH --job-name=eval_llava          # 作业名称
#SBATCH --nodes=2                      # 需要的节点数
#SBATCH --time=00:15:00                # 时间
#SBATCH --output=logs/job_%j.out       # 输出日志
#SBATCH --partition=gh                 # 分区

benchmark="vsibench"
model="llava_one_vision_qwen2_7b_ov_32f"
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_family="llava_onevision"
model_args="pretrained=LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-spann3r/checkpoint-150,\
conv_template=qwen_1_5,\
model_name=llava_qwen,\
max_frames_num=32"
export LMMS_EVAL_LAUNCHER="accelerate"

# 获取 SLURM 提供的分布式参数
WORLD_SIZE=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_PROCID
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=27500  # 自定义一个未被占用的端口

# 打印分布式参数
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# 修改启动命令
srun accelerate launch \
    --num_processes=$SLURM_JOB_NUM_NODES \
    --num_machines=$SLURM_JOB_NUM_NODES \
    --machine_rank=$SLURM_PROCID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark