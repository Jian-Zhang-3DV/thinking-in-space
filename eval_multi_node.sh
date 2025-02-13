#!/bin/bash

pretrained=${1:-"LLaVA-NeXT/checkpoints/LLaVA-Video-7B-Qwen2"}
use_lora=${2:-"false"}

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
max_frames_num=32

# 从pretrained路径中提取模型名称
model_name=$(basename ${pretrained})
model="llava_one_vision_${model_name}_ov_${max_frames_num}f"

output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_family="llava_onevision"
if [ "$use_lora" = "true" ]; then
    model_args="pretrained=${pretrained},\
    attn_implementation=flash_attention_2,\
    conv_template=qwen_1_5,\
    model_name=llava_qwen_lora,\
    model_base=LLaVA-NeXT/checkpoints/LLaVA-Video-7B-Qwen2,\
    max_frames_num=${max_frames_num}"
else
    model_args="pretrained=${pretrained},\
    attn_implementation=flash_attention_2,\
    conv_template=qwen_1_5,\
    model_name=llava_qwen,\
    max_frames_num=${max_frames_num}"
fi

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