#!/bin/bash

# Python 脚本路径
PYTHON_SCRIPT="lfd_mujoco.py"
# 算法列表
# ALGOS=("mybc" "demodice" "iswbc")
ALGOS=("metairl") 
# 环境列表
ENVS=("HalfCheetah-v2")
# 随机种子列表
# SEEDS=(2022 2023 2024 2025 2026)
SEEDS=(2023)
# GPU 设备列表
DEVICES=(1 1)

# 不完缺数据集和迷你传输数量
# IMPERFECT_DATASET_NAMES=("expert" "random")
# IMPERFECT_NUM_TRAJS=(400 1600)

IMPERFECT_DATASET_NAMES=("expert" "random")
IMPERFECT_NUM_TRAJS=(400 1600)

# 初始化 GPU 计数器
gpu_index=0

# 遍历所有算法、环境和随机种子
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # 分配 GPU 设备
            device=${DEVICES[gpu_index]}
            
            # 构建参数集并执行 Python 脚本
            echo "Running: algo=$algo, env=$env, seed=$seed, device=$device, dataset_names=${IMPERFECT_DATASET_NAMES[@]}, num_trajs=${IMPERFECT_NUM_TRAJS[@]}"
            CUDA_VISIBLE_DEVICES=$device python $PYTHON_SCRIPT \
                --algorithm "$algo" \
                --env_id "$env" \
                --seed "$seed" \
                --walpha 1.0 \
                --beta 0.0 \
                $(for ((i=0; i<${#IMPERFECT_DATASET_NAMES[@]}; i++)); do echo --imperfect_dataset_names "${IMPERFECT_DATASET_NAMES[i]}" --imperfect_num_trajs "${IMPERFECT_NUM_TRAJS[i]}"; done) &
            
            # 更新 GPU 计数器 (循环使用 0-3)
            gpu_index=$(( (gpu_index + 1) % 2 ))
        done
    done
done

# 等待所有背景任务完成
wait
echo "All processes finished."