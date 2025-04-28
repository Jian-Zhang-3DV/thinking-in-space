#!/bin/bash


benchmark="${BENCHMARK:-vsibench}"
model_family="vlm_3r"
model="${MODEL:-llava_video_7b_qwen2_lora_base}"
pretrained="checkpoints/${model}"
model_base="${MODEL_BASE:-}"

output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
model_args="pretrained=${pretrained},\
conv_template=qwen_1_5,\
max_frames_num=32"

if [ -n "$model_base" ]; then
    model_args="${model_args},model_base=${model_base}"
fi

export LMMS_EVAL_LAUNCHER="accelerate"

# 修改启动命令
accelerate launch \
    --num_processes=4 \
    -m lmms_eval \
    --model $model_family \
    --model_args $model_args \
    --tasks $benchmark \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $model \
    --output_path $output_path/$benchmark