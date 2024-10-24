#!/bin/bash

# Python 脚本路径
PYTHON_SCRIPT="lfd_mujoco.py"
# 算法列表
# ALGOS=("mybc" "demodice" "iswbc")
ALGOS=("ilmar" "metademodice" "metaiswbc") 
# 环境列表
ENVS=("Ant-v2" "Hopper-v2" "HalfCheetah-v2" "Walker2d-v2")
# 随机种子列表
# SEEDS=(2022 2023 2024 2025 2026)
SEEDS=(2022)
# GPU 设备列表
DEVICES=(0 1 2 3)

# 不完缺数据集和迷你传输数量
IMPERFECT_DATASET_NAMES=("full_replay")
IMPERFECT_NUM_TRAJS=(5000)

# 初始化 GPU 计数器
gpu_index=0

# 遍历所有算法、环境和随机种子
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for ((i=0; i<${#IMPERFECT_DATASET_NAMES[@]}; i++)); do
                # 分配 GPU 设备
                device=${DEVICES[gpu_index]}
                dataset_name=${IMPERFECT_DATASET_NAMES[i]}
                num_trajs=${IMPERFECT_NUM_TRAJS[i]}
                
                # 构建参数集并执行 Python 脚本
                echo "Running: algo=$algo, env=$env, seed=$seed, device=$device, dataset_name=$dataset_name, num_trajs=$num_trajs"
                CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT --algorithm "$algo" --imperfect_dataset_names "$dataset_name" --imperfect_num_trajs "$num_trajs" --env_id "$env" --seed "$seed" --walpha 1.0 --beta 0.0&  # 在背景执行
                # 更新 GPU 计数器 (循环使用 0-3)
                gpu_index=$(( (gpu_index + 1) % 4 ))
            done
        done
    done
done

# 等待所有背景任务完成
wait
echo "All processes finished."