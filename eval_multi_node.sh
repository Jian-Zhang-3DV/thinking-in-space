#!/bin/bash

# SLURM 环境下的分布式训练设置
export WORLD_SIZE=$SLURM_JOB_NUM_NODES  # 总节点数
export RANK=$SLURM_PROCID               # 当前节点的 rank，由 SLURM 自动分配
export LOCAL_RANK=$SLURM_LOCALID        # 本地 GPU ID，由 SLURM 自动分配
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # 主节点地址
export MASTER_PORT=29500   

# 打印环境变量进行检查
echo "分布式训练环境变量："
echo "WORLD_SIZE = $WORLD_SIZE"
echo "RANK = $RANK"
echo "LOCAL_RANK = $LOCAL_RANK"
echo "MASTER_ADDR = $MASTER_ADDR"
echo "MASTER_PORT = $MASTER_PORT"

benchmark="vsibench"
model="llava_one_vision_qwen2_7b_ov_64f"
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_family="llava_onevision"
model_args="pretrained=LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-spann3r/checkpoint-150,\
conv_template=qwen_1_5,\
model_name=llava_qwen,\
max_frames_num=64"
export LMMS_EVAL_LAUNCHER="accelerate"

accelerate launch \
    --num_machines=$WORLD_SIZE \
    --num_processes=$WORLD_SIZE \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --multi_gpu \
    -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark