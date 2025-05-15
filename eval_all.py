import subprocess
import os
import sys
import time

# --- Configuration ---
# Please replace these lists with your actual model and benchmark names
MODELS_and_MODEL_BASES = [
    # ("llava_video_7b_qwen2_lora_base", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    # ("llava_video_7b_qwen2_last_hidden_state_and_cam_token", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    # ("llava_video_7b_qwen2_lora_last_hidden_state_and_cam_token", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    # ("LLaVA-Video-7B-Qwen2", ""),
    # ("llava_video_7b_qwen2_05_11_lora_base_cam_obj_abs_dist_standalone/checkpoint-1200", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    # ("llava_video_7b_qwen2_05_13_lora_cut3r_all_tokens_cross_attn_cam_obj_abs_dist_rel_dir/checkpoint-1671", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"), 
    ("llava_video_7b_qwen2_05_15_lora_cut3r_all_tokens_cross_attn_vstibench/checkpoint-447", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"), 
    ("llava_video_7b_qwen2_05_15_lora_base_vstibench_resample/checkpoint-447", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"), 
]
BENCHMARKS = [
    "vstibench",
    # "vsibench",
]
# Path to the script to execute
SCRIPT_PATH = "./eval_single_node.sh"
# Directory to store log files
LOG_DIR = "./evaluation_logs"

# --- Execution ---
def run_evaluation(benchmark, model, model_base):
    """Runs the evaluation script with the given benchmark and model, redirecting output to a log file."""
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Sanitize model name for filename
    safe_model_name = model.replace("/", "_") # Replace slashes which are invalid in filenames
    log_filename = f"{benchmark}_{safe_model_name}.log"
    log_filepath = os.path.join(LOG_DIR, log_filename)

    print(f"--- Running Benchmark: {benchmark}, Model: {model}, Model Base: {model_base} ---", flush=True)
    print(f"--- Logging output to: {log_filepath} ---", flush=True)

    # Create a copy of the current environment variables
    env = os.environ.copy()
    # Set the specific variables for the script
    env["BENCHMARK"] = benchmark
    env["MODEL"] = model
    env["MODEL_BASE"] = model_base

    try:
        # Open the log file in append mode
        with open(log_filepath, 'a') as log_file:
            # Execute the script, redirecting stdout and stderr to the log file
            process = subprocess.run(
                ["bash", SCRIPT_PATH],
                env=env,
                check=True, # Raise an exception if the script returns a non-zero exit code
                stdout=log_file, # Redirect stdout to the file
                stderr=log_file, # Redirect stderr to the same file
                text=True # Decode stdout/stderr as text (though it goes to file now)
            )
        print(f"Successfully completed Benchmark: {benchmark}, Model: {model}")
    except subprocess.CalledProcessError as e:
        # Log error messages to the main console as well as the specific log file
        error_message = f"""
*** Error running Benchmark: {benchmark}, Model: {model} ***
Return code: {e.returncode}
Error details might be in: {log_filepath}
Check the log file for stdout/stderr from the script.
"""
        print(error_message, file=sys.stderr)
        # Optionally, append error info to the log file if it wasn't captured (e.g., if file open failed)
        try:
             with open(log_filepath, 'a') as log_file:
                 log_file.write("\n--- Subprocess Error Info ---\n")
                 log_file.write(f"Return code: {e.returncode}\n")
                 if e.stdout:
                     log_file.write("Captured Stdout:\n")
                     log_file.write(e.stdout + "\n")
                 if e.stderr:
                     log_file.write("Captured Stderr:\n")
                     log_file.write(e.stderr + "\n")
        except Exception as log_err:
             print(f"*** Additionally, failed to write error details to log file {log_filepath}: {log_err} ***", file=sys.stderr)

    except FileNotFoundError:
        print(f"*** Error: Script not found at {SCRIPT_PATH} ***", file=sys.stderr)
    except Exception as e:
        print(f"*** An unexpected error occurred for Benchmark: {benchmark}, Model: {model}: {e} ***", file=sys.stderr)
        # Also try to log this unexpected error
        try:
             with open(log_filepath, 'a') as log_file:
                 log_file.write(f"\n--- Unexpected Python Error ---\n{str(e)}\n")
        except Exception as log_err:
             print(f"*** Additionally, failed to write unexpected error to log file {log_filepath}: {log_err} ***", file=sys.stderr)
    finally:
        print(f"--- Finished Benchmark: {benchmark}, Model: {model} ---", flush=True)

def main():
    if not os.path.exists(SCRIPT_PATH):
        print(f"Error: Evaluation script '{SCRIPT_PATH}' not found.", file=sys.stderr)
        sys.exit(1)
        
    if not os.access(SCRIPT_PATH, os.X_OK):
        print(f"Error: Evaluation script '{SCRIPT_PATH}' is not executable. Attempting to chmod +x...", file=sys.stderr)
        try:
            os.chmod(SCRIPT_PATH, 0o755)
            print(f"Successfully made '{SCRIPT_PATH}' executable.")
        except Exception as e:
             print(f"Failed to make script executable: {e}", file=sys.stderr)
             sys.exit(1)

    print("Starting all evaluations...")
    total_runs = len(BENCHMARKS) * len(MODELS_and_MODEL_BASES)
    current_run = 0
    for benchmark in BENCHMARKS:
        for model, model_base in MODELS_and_MODEL_BASES:
            current_run += 1
            print(f"\n=== Evaluation Run {current_run}/{total_runs} ===")
            run_evaluation(benchmark, model, model_base)

            # wait for 10 seconds
            time.sleep(10)
            
    print("\nAll evaluations finished.")

if __name__ == "__main__":
    main()
