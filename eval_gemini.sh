# bash evaluate_all_in_one.sh --model all --num_processes 8 --benchmark vsibench
# bash evaluate_all_in_one.sh --model all --num_processes 2 --benchmark vstrbench

#!/bin/bash

# Define the tasks as an array
# tasks=(
#     "vstrbench"
# )
task="vstibench"

# Output file
output_file="all_tasks_output_gemini-2.5.txt"

# Clear the output file at the start
> "$output_file"

# Loop through each task and run the evaluation command, appending output to the same file
echo "Evaluating task: $task" | tee -a "$output_file"
# bash evaluate_all_in_one.sh --model gemini_2p0_flash_exp --num_processes 1 --benchmark "$task" >> "$output_file"

bash evaluate_all_in_one.sh --model gemini-2.5-flash-preview-04-17 --num_processes 8 --benchmark $task >> "$output_file"

bash evaluate_all_in_one.sh --model gemini-2.5-pro-preview-05-06 --num_processes 8 --benchmark $task >> "$output_file"


echo "All tasks have been evaluated and their outputs saved in $output_file."