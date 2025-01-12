#!/bin/bash

# 添加命令行参数检查
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_step>"
    exit 1
fi

benchmark="vsibench"
checkpoint_step=$1  # 从命令行参数获取 checkpoint_step
model="llava_one_vision_qwen2_7b_ov_32f_$checkpoint_step"
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_family="llava_onevision"
model_args="pretrained=LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-spann3r/checkpoint-$checkpoint_step,\
conv_template=qwen_1_5,\
model_name=llava_qwen,\
max_frames_num=32"

export LMMS_EVAL_LAUNCHER="accelerate"

# 修改启动命令
accelerate launch \
    --num_processes=1 \
    -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark