#!/bin/bash

# GPU 固定为 1
GPU_NUM=1

# LRO 配置列表，每项为 [L, R, O]
LRO_CONFIGS=(
    # "0 0 1"
    # "1 0 0"
    # "0 1 0"
    # "1 1 1"
)

# 通用运行函数
run_benchmark() {
    local MODEL=$1
    local BS_START=$2
    local BS_END=$3
    local BS_STEP=$4

    for BS in $(seq ${BS_START} ${BS_STEP} ${BS_END})
    do
        for LRO in "${LRO_CONFIGS[@]}"
        do
            # 解析 LRO 配置
            L=$(echo $LRO | cut -d ' ' -f 1)
            R=$(echo $LRO | cut -d ' ' -f 2)
            O=$(echo $LRO | cut -d ' ' -f 3)

            # 提示当前运行的任务信息
            echo "Running GMLake Pro Max: Model=${MODEL}, BS=${BS}, L=${L}, R=${R}, O=${O}"

            # 调用 finetune.sh 脚本，启用 GMLake
            bash training_scripts/finetune.sh ${GPU_NUM} ${BS} ${MODEL} True ${L} ${R} ${O}
        done
    done
}

# # 模型 1: facebook/opt-125m
# run_benchmark "facebook/opt-125m" 1 1 1

# # 模型 2: facebook/opt-350m
# run_benchmark "facebook/opt-350m" 16 64 16

# # 模型 3: facebook/opt-1.3b
# run_benchmark "facebook/opt-1.3b" 8 32 8
