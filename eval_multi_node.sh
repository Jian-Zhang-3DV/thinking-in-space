#!/bin/bash

pretrained=$1
benchmark=$2
model_base=$3

# Check if LoRA is used and model_base is provided
if [[ "$pretrained" == *lora* ]] && [ -z "$model_base" ]; then
    echo "Error: model_base (third argument) is required when 'lora' is in the pretrained path." >&2
    exit 1
fi

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

# hard code
max_frames_num=32
model_family="vlm_3r"
# Extract model name from pretrained path for log suffix (handles checkpoint dirs)
# Remove trailing slash if present
cleaned_path=${pretrained%/}
last_component=$(basename "$cleaned_path")
if [[ "$last_component" == checkpoint-* ]]; then
    parent_dir=$(dirname "$cleaned_path")
    parent_name=$(basename "$parent_dir")
    log_suffix_model_name="${parent_name}_${last_component}"
else
    log_suffix_model_name="$last_component"
fi

output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")

# 基础参数
model_args="pretrained=${pretrained},\
attn_implementation=flash_attention_2,\
conv_template=qwen_1_5,\
max_frames_num=${max_frames_num}"

# 如果提供了 model_base，则添加到参数中
if [ -n "$3" ]; then
    model_args="${model_args},model_base=${model_base}"
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
    --log_samples_suffix $log_suffix_model_name \
    --output_path $output_path/$benchmark