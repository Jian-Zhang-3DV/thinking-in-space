import subprocess
import os
import sys
import time
import argparse
import multiprocessing

# --- Configuration ---
# Define model configurations including nodes and time limits
# Format: (model_name, model_base, nodes, time_limit)
MODELS_and_CONFIGS = [
    # Example:
    # ("model_name_1", "base_model_1", 4, "01:00:00"),
    # ("model_name_2", "base_model_2", 8, "02:30:00"),
    # ("llava_video_7b_qwen2_04_30_lora_base/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:00:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_04_30_lora_last_hidden_state/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 16, "01:30:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_base_mlp/checkpoint-700", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:00:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_base_mlp/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:00:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_base_mlp/checkpoint-2100", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:00:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_patch_tokens_2_layer_cross_attn/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_patch_tokens_2_layer_cross_attn/checkpoint-700", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_01_lora_patch_tokens_2_layer_cross_attn/checkpoint-2100", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    # ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_mlp/checkpoint-700", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_mlp/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_mlp/checkpoint-2100", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_2_layers_mlp/checkpoint-700", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_2_layers_mlp/checkpoint-1400", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
    ("llava_video_7b_qwen2_05_02_lora_patch_tokens_cross_attn_2_layers_mlp/checkpoint-2100", "lmms-lab/LLaVA-NeXT-Video-7B-Qwen2", 8, "01:30:00"), # Added example nodes and time
]
BENCHMARKS = [
    # "vstrbench",
    "vsibench",
]
# Path to the SLURM script template
SLURM_SCRIPT_PATH = "./run_eval.slurm"
# Directory to store log files (optional, SLURM handles its own logs)
LOG_DIR = "./slurm_submission_logs" # Log for this submission script, not the SLURM jobs themselves
MODEL_BASE_DIR = "$SCRATCH/work_dirs_auto_eval"

# --- New Configuration for Pre-check ---
WAIT_CHECK_INTERVAL = 60  # seconds (How often to check for directory existence)
SIZE_CHECK_INTERVAL = 30  # seconds (How often to check directory size)
STABILITY_DURATION = 120 # seconds (How long size must be stable before proceeding)
MAX_WAIT_TIME = 86400 # seconds (Maximum time to wait for directory to exist, e.g., 1 day)


# --- Helper Function ---
def get_dir_size(directory):
    """Calculates the total size of all files in a directory recursively."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is symbolic link
                if not os.path.islink(fp):
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError as e:
                        print(f"Warning: Could not get size of file {fp}: {e}", file=sys.stderr, flush=True)
    except FileNotFoundError:
         print(f"Warning: Directory {directory} not found during size calculation.", file=sys.stderr, flush=True)
         return 0 # Treat as 0 size if not found during calculation phase
    except OSError as e:
        print(f"Warning: Error walking directory {directory}: {e}", file=sys.stderr, flush=True)
        return -1 # Indicate an error occurred during walk
    return total_size


# --- Functions ---
def submit_slurm_job(benchmark, model, model_base, nodes, time_limit): # Added nodes and time_limit here
    """Checks for model directory stability and then submits a SLURM job."""
    # Ensure log directory for this script exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Sanitize model name for job name (optional, but can be helpful)
    safe_model_name = model.replace("/", "_")
    job_name = f"eval_{benchmark}_{safe_model_name}"
    # Log file for the sbatch *submission* itself, AND the job output/error
    submission_log_filename = f"submit_{job_name}.log"
    submission_log_filepath = os.path.join(LOG_DIR, submission_log_filename)

    # Expand the environment variable in the base directory path
    expanded_model_base_dir = os.path.expandvars(MODEL_BASE_DIR)
    # Construct the full path to the pretrained model
    pretrained_path = os.path.join(expanded_model_base_dir, model)

    # --- Pre-submission Checks ---
    print(f"--- Pre-check for: {pretrained_path} ---", flush=True)

    # 1. Wait for the directory to exist, with a timeout
    wait_start_time = time.time()
    directory_exists = False
    while time.time() - wait_start_time < MAX_WAIT_TIME:
        if os.path.exists(pretrained_path):
            directory_exists = True
            print(f"Directory found: {pretrained_path}. Checking for size stability...", flush=True)
            break
        else:
            elapsed_wait = time.time() - wait_start_time
            # Check remaining time before sleeping
            remaining_time = MAX_WAIT_TIME - elapsed_wait
            wait_this_interval = min(WAIT_CHECK_INTERVAL, remaining_time)
            if wait_this_interval <= 0: # No more time left to wait
                 break
            print(f"Directory not found: {pretrained_path}. Waiting {wait_this_interval:.1f}s... (Elapsed: {elapsed_wait:.1f}/{MAX_WAIT_TIME}s)", flush=True)
            time.sleep(wait_this_interval) # Sleep for the calculated interval

    if not directory_exists:
        error_message = f"*** Error: Timeout waiting for directory {pretrained_path} to exist after {MAX_WAIT_TIME}s. Skipping job submission for Benchmark: {benchmark}, Model: {model} ***"
        print(error_message, file=sys.stderr, flush=True)
        try:
            # Append timeout error to the log file designated for this job's submission attempt
            with open(submission_log_filepath, 'a') as log_file:
                log_file.write(f"--- Pre-check Error ---\n{error_message}\n")
                log_file.flush() # Ensure error is written
        except Exception as log_err:
            print(f"*** Additionally, failed to write timeout error to log file {submission_log_filepath}: {log_err} ***", file=sys.stderr)
        # Skip the rest of the submission process for this job
        print(f"--- Skipping submission for Benchmark: {benchmark}, Model: {model} due to timeout ---", flush=True)
        return # Exit this function call and proceed to the next job in main()


    # 2. Wait for directory size to stabilize
    last_size = -1 # Initialize with a value that won't match the first check
    stable_start_time = None
    while True:
        current_size = get_dir_size(pretrained_path)
        current_time = time.time()

        if current_size == -1: # Error getting size
             print(f"Error calculating size for {pretrained_path}. Skipping stability check and proceeding.", file=sys.stderr, flush=True)
             # Optionally log this error to the submission log file before proceeding
             try:
                 with open(submission_log_filepath, 'a') as log_file:
                     log_file.write(f"""--- Pre-check Warning ---
Failed to calculate directory size accurately for {pretrained_path}. Proceeding without stability confirmation.
""")
             except Exception as log_err:
                 print(f"*** Additionally, failed to write size check error to log file {submission_log_filepath}: {log_err} ***", file=sys.stderr)
             break # Proceed with submission despite error

        print(f"  - Current size: {current_size} bytes", flush=True)

        if current_size == last_size:
            if stable_start_time is None:
                # Start timer only if size is non-zero or if we allow stability check for zero size
                # (currently allowing zero size stability)
                stable_start_time = current_time # Start timer when size first matches previous check
                print(f"  - Size ({current_size} bytes) hasn't changed since last check. Starting stability timer.", flush=True)
            else:
                elapsed_stable_time = current_time - stable_start_time
                print(f"  - Size stable for {elapsed_stable_time:.1f}s (target: {STABILITY_DURATION}s).", flush=True)
                if elapsed_stable_time >= STABILITY_DURATION:
                    print(f"Directory size has been stable for {STABILITY_DURATION}s. Assuming model saving is complete.", flush=True)
                    break # Size is stable
        else:
            # Size changed, or this is the first check (last_size was -1)
            if last_size != -1: # Avoid printing reset message on the very first check
                 print(f"  - Size changed ({last_size} -> {current_size}). Resetting stability timer.", flush=True)
            else:
                 print(f"  - Initial size check: {current_size} bytes.", flush=True)
            last_size = current_size
            stable_start_time = None # Reset stability timer

        print(f"Waiting {SIZE_CHECK_INTERVAL}s before next size check...", flush=True)
        time.sleep(SIZE_CHECK_INTERVAL)
    # --- End of Pre-submission Checks ---


    print(f"--- Submitting SLURM Job ---", flush=True)
    print(f"Benchmark: {benchmark}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Model Base: {model_base}", flush=True)
    print(f"Pretrained Path (expanded): {pretrained_path}", flush=True) # Print the expanded path
    print(f"Nodes: {nodes}", flush=True) # Now specific to this job
    print(f"Time Limit: {time_limit}", flush=True) # Now specific to this job
    print(f"SLURM Script: {SLURM_SCRIPT_PATH}", flush=True)
    # Update print statement: This file will contain both submission and job logs
    print(f"Combined Submission and Job Log: {submission_log_filepath}", flush=True)

    # Construct the sbatch command
    # Note: We override SBATCH directives from the script using command-line options
    sbatch_command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--nodes={nodes}", # Use job-specific nodes
        f"--time={time_limit}", # Use job-specific time limit
        f"--output={submission_log_filepath}", # Send job stdout to the submission log file
        f"--error={submission_log_filepath}",  # Send job stderr to the submission log file
        "--open-mode=append",             # Append job output/error to the file
        # Add other SBATCH directives if needed, e.g., partition, account
        # f"--partition=your_partition", # Example
        SLURM_SCRIPT_PATH, # The script to run
        # Arguments passed to the SLURM script (run_eval.slurm)
        pretrained_path,        # Use the expanded path
        benchmark,              # benchmark argument for run_eval.slurm
        model_base              # model_base argument for run_eval.slurm
    ]

    try:
        # Open the submission log file (in write mode, will overwrite if exists)
        with open(submission_log_filepath, 'w') as log_file:
            cmd_str = ' '.join(sbatch_command)
            print(f"Executing command: {cmd_str}", flush=True)
            log_file.write(f"Executing command: {cmd_str}\n")
            log_file.flush() # Ensure header is written

            # Execute the sbatch command
            process = subprocess.run(
                sbatch_command,
                check=True, # Raise an exception if sbatch fails
                stdout=subprocess.PIPE, # Capture sbatch output (like job ID)
                stderr=subprocess.PIPE, # Capture sbatch errors separately
                text=True
            )
            # Log successful submission output (usually contains the Job ID)
            log_file.write("\n--- sbatch stdout ---\n")
            captured_stdout = process.stdout or "No stdout captured." # Handle potential None
            log_file.write(captured_stdout + "\n") # Add newline for log readability
            log_file.flush() # Ensure stdout is written to log

            print(f"Successfully submitted job for Benchmark: {benchmark}, Model: {model}")
            # Check if stdout was captured before stripping
            if process.stdout:
                 print(f"sbatch output: {process.stdout.strip()}") # Show Job ID on console
            else:
                 print("sbatch output: (Not captured or empty)")

    except subprocess.CalledProcessError as e:
        # Define the error message using an f-string
        error_message = (
            f"*** Error submitting SLURM job for Benchmark: {benchmark}, Model: {model} ***\n"
            f"Return code: {e.returncode}\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Stderr:\n{e.stderr}\n"
            f"Check submission log for details: {submission_log_filepath}\n"
        )
        print(error_message, file=sys.stderr)
        # Ensure the error is also in the submission log
        try:
             with open(submission_log_filepath, 'a') as log_file:
                 log_file.write("\n--- sbatch stderr ---\n")
                 log_file.write(e.stderr or "No stderr captured.")
        except Exception as log_err:
             print(f"*** Additionally, failed to write sbatch error details to log file {submission_log_filepath}: {log_err} ***", file=sys.stderr)

    except FileNotFoundError:
        print(f"*** Error: sbatch command not found, or SLURM script not found at {SLURM_SCRIPT_PATH} ***", file=sys.stderr)

    except Exception as e:
        print(f"*** An unexpected error occurred during submission for Benchmark: {benchmark}, Model: {model}: {e} ***", file=sys.stderr)
        # Also try to log this unexpected error
        try:
             with open(submission_log_filepath, 'a') as log_file:
                 log_file.write(f"\n--- Unexpected Python Error During Submission ---\n{str(e)}\n")
        except Exception as log_err:
             print(f"*** Additionally, failed to write unexpected error to submission log file {submission_log_filepath}: {log_err} ***", file=sys.stderr)

    finally:
        print(f"--- Finished processing submission for Benchmark: {benchmark}, Model: {model} ---", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Submit multiple SLURM evaluation jobs with individual configurations in parallel.")
    parser.add_argument(
        '--parallel-jobs', 
        type=int, 
        default=4, 
        help='Number of job checks/submissions to run in parallel.'
    )
    # Removed --nodes and --time arguments as they are per-job now
    args = parser.parse_args()

    if not os.path.exists(SLURM_SCRIPT_PATH):
        print(f"Error: SLURM script '{SLURM_SCRIPT_PATH}' not found.", file=sys.stderr)
        sys.exit(1)

    # --- Prepare Tasks --- 
    tasks_to_run = []
    for benchmark in BENCHMARKS:
        for model, model_base, nodes, time_limit in MODELS_and_CONFIGS:
            tasks_to_run.append((benchmark, model, model_base, nodes, time_limit))
    
    total_submissions = len(tasks_to_run)
    print(f"Found {total_submissions} total jobs to process.")
    print(f"Running up to {args.parallel_jobs} checks/submissions in parallel...")

    # --- Execute Tasks in Parallel --- 
    # Ensure log directory exists before starting workers
    os.makedirs(LOG_DIR, exist_ok=True) 

    if total_submissions > 0:
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=args.parallel_jobs) as pool:
            # Use starmap to pass arguments from each tuple in tasks_to_run to submit_slurm_job
            # This will block until all tasks are complete
            pool.starmap(submit_slurm_job, tasks_to_run)
        print("--- All parallel submission processes finished. --- ")
    else:
        print("No jobs found in configuration. Exiting.")


if __name__ == "__main__":
    # Need this guard for multiprocessing on some OSes
    main() 