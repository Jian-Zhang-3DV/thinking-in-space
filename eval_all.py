import subprocess
import os
import sys
import time

# --- Configuration ---
# Please replace these lists with your actual model and benchmark names
MODELS_and_MODEL_BASES = [
    ("llava_video_7b_qwen2_lora_base", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    ("llava_video_7b_qwen2_last_hidden_state_and_cam_token", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    ("llava_video_7b_qwen2_lora_last_hidden_state_and_cam_token", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2"),
    ("lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", ""),
]
BENCHMARKS = [
    "vsibench",
    "vstrbench",
]
# Path to the script to execute
SCRIPT_PATH = "./eval_single_node.sh"

# --- Execution ---
def run_evaluation(benchmark, model, model_base):
    """Runs the evaluation script with the given benchmark and model."""
    print(f"--- Running Benchmark: {benchmark}, Model: {model}, Model Base: {model_base} ---", flush=True)
    
    # Create a copy of the current environment variables
    env = os.environ.copy()
    # Set the specific variables for the script
    env["BENCHMARK"] = benchmark
    env["MODEL"] = model
    env["MODEL_BASE"] = model_base
    try:
        # Execute the script
        process = subprocess.run(
            ["bash", SCRIPT_PATH],
            env=env,
            check=True, # Raise an exception if the script returns a non-zero exit code
            stdout=subprocess.PIPE, # Capture standard output
            stderr=subprocess.PIPE, # Capture standard error
            text=True # Decode stdout/stderr as text
        )
        print(f"Successfully completed Benchmark: {benchmark}, Model: {model}")
        print("Output:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"*** Error running Benchmark: {benchmark}, Model: {model} ***", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("Stderr:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("Stdout:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
    except FileNotFoundError:
        print(f"*** Error: Script not found at {SCRIPT_PATH} ***", file=sys.stderr)
    except Exception as e:
        print(f"*** An unexpected error occurred for Benchmark: {benchmark}, Model: {model}: {e} ***", file=sys.stderr)
    finally:
        print(f"--- Finished Benchmark: {benchmark}, Model: {model} ---
", flush=True)

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
    total_runs = len(BENCHMARKS) * len(MODELS)
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
