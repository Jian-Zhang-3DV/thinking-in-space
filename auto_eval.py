#!/usr/bin/env python3
import os
import time
from pathlib import Path

def get_dir_size(path):
    """获取文件夹的总大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = Path(dirpath) / f
            if not fp.is_symlink():  # 跳过符号链接
                total_size += fp.stat().st_size
    return total_size

def is_checkpoint_complete(checkpoint_path, check_interval=60, max_checks=5):
    """
    检查checkpoint文件夹是否完整：
    通过多次检查文件夹大小是否保持稳定来判断写入是否完成
    
    Args:
        checkpoint_path: checkpoint文件夹路径
        check_interval: 每次检查的间隔时间（秒）
        max_checks: 连续稳定的次数要求
    """
    previous_size = get_dir_size(checkpoint_path)
    stable_count = 0
    
    for _ in range(max_checks):
        time.sleep(check_interval)
        current_size = get_dir_size(checkpoint_path)
        
        if current_size == previous_size and current_size > 0:
            stable_count += 1
        else:
            stable_count = 0
        
        previous_size = current_size
        
        # 如果连续多次大小保持稳定，认为写入已完成
        if stable_count >= 2:  # 至少需要连续3次检查大小都相同
            return True
            
    return False

def find_unevaluated_checkpoints(parent_dir, work_dir, conda_env, interval=600):  # Default 10 minutes interval
    """
    Recursively monitor a parent directory for checkpoint folders without .log files
    
    Args:
        parent_dir (str): Path to parent directory to monitor
        work_dir (str): Path to the project directory
        conda_env (str): Name of the conda environment to activate
        interval (int): Time interval in seconds between checks
    """
    while True:
        unevaluated = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(parent_dir):
            # Find checkpoint directories
            checkpoint_dirs = [d for d in dirs if d.startswith('checkpoint')]
            
            for checkpoint_dir in checkpoint_dirs:
                checkpoint_path = Path(root) / checkpoint_dir
                
                # Check if directory contains .log files
                has_log = False
                for f in os.listdir(checkpoint_path):
                    if f.endswith('.log'):
                        has_log = True
                        break
                
                # 只有当没有log文件且checkpoint完整时才添加到待评估列表
                if not has_log and is_checkpoint_complete(checkpoint_path):
                    unevaluated.append(str(checkpoint_path))
        
        if unevaluated:
            print(f"Found {len(unevaluated)} complete and unevaluated checkpoints:")
            for checkpoint_path in unevaluated:
                print(f"  {checkpoint_path}")
                # 构建sbatch命令，添加conda环境激活和工作目录设置
                cmd = (
                    f"cd {work_dir} && "
                    f"source ~/.bashrc && "
                    f"conda activate {conda_env} && "
                    f"sbatch -N 16 -p gh -t 00:20:00 "
                    f"--output={checkpoint_path}/eval.log "  # 修改日志输出路径
                    f"run_eval.slurm {checkpoint_path} "
                    f"llava_qwen_lora LLaVA-NeXT/checkpoints/LLaVA-Video-7B-Qwen2"
                )
                os.system(cmd)  # 执行命令
        
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
