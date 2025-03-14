#!/usr/bin/env python3
import os
import time
from pathlib import Path

def get_dir_size(path):
    """Get total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = Path(dirpath) / f
            if not fp.is_symlink():  # Skip symbolic links
                total_size += fp.stat().st_size
    return total_size

def is_checkpoint_complete(checkpoint_path, check_interval=60, max_checks=5):
    """
    Check if checkpoint directory is complete by monitoring its size stability
    
    Args:
        checkpoint_path: Path to checkpoint directory
        check_interval: Time interval between checks in seconds
        max_checks: Maximum number of checks to perform
    
    Returns:
        bool: True if directory size remains stable, False otherwise
    """
    print(f"\nChecking stability for checkpoint: {checkpoint_path}")
    previous_size = get_dir_size(checkpoint_path)
    print(f"Initial size: {previous_size / 1024 / 1024:.2f} MB")
    stable_count = 0
    
    for check_num in range(max_checks):
        print(f"Stability check {check_num + 1}/{max_checks}, waiting {check_interval}s...")
        time.sleep(check_interval)
        current_size = get_dir_size(checkpoint_path)
        
        print(f"Current size: {current_size / 1024 / 1024:.2f} MB")
        if current_size == previous_size and current_size > 0:
            stable_count += 1
            print(f"Size remained stable. Stable count: {stable_count}/3")
        else:
            stable_count = 0
            print(f"Size changed. Resetting stable count to 0")
        
        previous_size = current_size
        
        # Consider complete if size remains stable for 3 consecutive checks
        if stable_count >= 2:
            print("Checkpoint appears complete (size stable for 3 consecutive checks)")
            return True
            
    print("Checkpoint not yet complete (size still changing)")
    return False

def find_unevaluated_checkpoints(parent_dir, work_dir, conda_env, interval=600):
    """
    Recursively monitor a directory for unevaluated checkpoint folders
    
    Args:
        parent_dir: Directory to monitor for checkpoints
        work_dir: Project directory containing evaluation scripts
        conda_env: Conda environment name for evaluation
        interval: Time interval between monitoring cycles in seconds
    """
    while True:
        print(f"\n=== Starting new monitoring cycle at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        unevaluated = []
        
        print(f"Scanning directory: {parent_dir}")
        for root, dirs, files in os.walk(parent_dir):
            checkpoint_dirs = [d for d in dirs if d.startswith('checkpoint')]
            
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_path = Path(root) / checkpoint_dir
                print(f"\nProcessing: {checkpoint_path}")
                
                # Check for existing log files
                has_log = False
                for f in os.listdir(checkpoint_path):
                    if f.endswith('.log'):
                        has_log = True
                        print(f"Found existing log file: {f}")
                        break
                
                if has_log:
                    print("Skipping: checkpoint already evaluated")
                    continue
                
                # Check if checkpoint is complete
                if is_checkpoint_complete(checkpoint_path):
                    unevaluated.append(str(checkpoint_path))
                    print("Added to evaluation queue")
        
        if unevaluated:
            print(f"\nFound {len(unevaluated)} checkpoints ready for evaluation:")
            for checkpoint_path in unevaluated:
                print(f"\nSubmitting evaluation job for: {checkpoint_path}")
                cmd = (
                    f"cd {work_dir} && "
                    f"source ~/.bashrc && "
                    f"conda activate {conda_env} && "
                    f"sbatch -N 16 -p gh -t 00:20:00 "
                    f"--output={checkpoint_path}/eval.log "
                    f"run_eval.slurm {checkpoint_path} "
                    f"llava_qwen_lora LLaVA-NeXT/checkpoints/LLaVA-Video-7B-Qwen2"
                )
                print("Executing command:", cmd)
                os.system(cmd)
                print("Job submitted")
        else:
            print("\nNo checkpoints ready for evaluation")
        
        print(f"\nSleeping for {interval} seconds...")
        time.sleep(interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", help="Parent directory to monitor")
    parser.add_argument("--work-dir", required=True, help="Directory containing run_eval.slurm")
    parser.add_argument("--conda-env", required=True, help="Conda environment name")
    parser.add_argument("--interval", type=int, default=600,
                       help="Check interval in seconds (default: 600)")
    args = parser.parse_args()
    
    find_unevaluated_checkpoints(
        args.parent_dir,
        args.work_dir,
        args.conda_env,
        args.interval
    )
