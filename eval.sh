# bash evaluate_all_in_one.sh --model all --num_processes 8 --benchmark vsibench
# export CUDA_VISIBLE_DEVICES=0,2
# bash evaluate_all_in_one.sh --model all --num_processes 2 --benchmark vstrbench

#!/bin/bash

# Define the tasks as an array
# tasks=(
#     "vstrbench"
# )
task="vstrbench"

# Output file
output_file="all_tasks_output.txt"

# Clear the output file at the start
> "$output_file"

# Loop through each task and run the evaluation command, appending output to the same file
echo "Evaluating task: $task" | tee -a "$output_file"
bash evaluate_all_in_one.sh --model internvl2_40b_8f --num_processes 2 --benchmark "$task" >> "$output_file"

bash evaluate_all_in_one.sh --model llava_one_vision_qwen2_72b_ov_32f --num_processes 2 --benchmark "$task" >> "$output_file"

bash evaluate_all_in_one.sh --model llava_next_video_72b_qwen2_32f --num_processes 2 --benchmark "$task" >> "$output_file"

echo "All tasks have been evaluated and their outputs saved in $output_file."