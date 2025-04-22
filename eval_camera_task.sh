#!/bin/bash

# 设置环境变量
export LMMS_EVAL_LAUNCHER="python"
export CUDA_VISIBLE_DEVICES="3"

# 执行 Python 模块并传递参数
# 使用反斜杠 \ 来换行以提高可读性
python -m lmms_eval \
    --model llava_vid \
    --model_args "pretrained=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=32" \
    --tasks camera_tasks \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_next_video_7b_qwen2_32f \
    --output_path logs/debug/camera_tasks

# 可选：取消设置环境变量（如果脚本结束后不需要它们）
# unset LMMS_EVAL_LAUNCHER
# unset CUDA_VISIBLE_DEVICES