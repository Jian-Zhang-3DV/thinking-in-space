#!/bin/bash

# 设置分布式相关参数
export WORLD_SIZE=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_PROCID
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=27500

# 打印分布式参数（用于调试）
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

benchmark="vsibench"
model="llava_one_vision_qwen2_7b_ov_32f"
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_family="llava_onevision"
model_args="pretrained=LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-spann3r/checkpoint-150,\
conv_template=qwen_1_5,\
model_name=llava_qwen,\
max_frames_num=32"
export LMMS_EVAL_LAUNCHER="accelerate"

accelerate launch \
    --num_processes=$WORLD_SIZE \
    --num_machines=$WORLD_SIZE \
    --machine_rank=$NODE_RANK \
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