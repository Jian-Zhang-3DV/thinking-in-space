def calculate_average_token_cost(filepath):
    total_count = 0
    total_cost = 0

    # 打开文件并逐行读取
    with open(filepath, 'r') as file:
        for line in file:
            # 检查是否为total_token_count行
            if "total_token_count:" in line:
                parts = line.split()
                # 确保行格式正确并有足够的部分来解析
                if len(parts) >= 2:
                    try:
                        # 将token计数和花销加到总数
                        token_count = int(parts[1])
                        total_count += token_count
                        # 假设花销在行的某个位置，需要根据实际情况调整
                        cost = float(parts[-1])  # 假设花销在最后一个位置
                        total_cost += cost * token_count
                    except ValueError:
                        # 如果转换失败，跳过这一行
                        print(f"Skipping line: {line.strip()} due to parsing error")
                        continue

    # 计算平均花销
    if total_count > 0:
        average_cost = total_cost / total_count
        print(f"Average Token Cost: {average_cost}")
    else:
        print("No valid token count lines found.")

# 路径可能需要根据实际情况调整
file_path = '/rhome/jzhan744/bigdata/thinking-in-space/all_tasks_output_gemini_1p5_flash.txt'
calculate_average_token_cost(file_path)