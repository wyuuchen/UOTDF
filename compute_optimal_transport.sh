#!/bin/bash

SRC_TYPES=("medium" "medium-replay" "medium-expert")
TAR_TYPES=("medium" "medium-expert" "expert")
ENV_NAMES=("ant" "ant-morph" "ant-gravity" "halfcheetah" "halfcheetah-morph" "halfcheetah-gravity" "hopper" "hopper-morph" "hopper-gravity" "walker2d" "walker2d-morph" "walker2d-gravity")

EPSILON=0.01
LAMBDA_SRC=0.05
LAMBDA_TAR=0.5
FILTER_THRESH=1.0
SEED=0
# 指定 GPU (例如使用 0 号卡)
export CUDA_VISIBLE_DEVICES=0

# 日志存放的主目录
LOG_DIR="./run_ot"
mkdir -p "$LOG_DIR"

# ================= 执行循环 =================

echo "Starting Batch Experiments..."
echo "Total Configurations: $(( ${#ROBOTS[@]} *  ${#SRC_TYPES[@]} * ${#TAR_TYPES[@]} ))"

for ENV_NAME in "${ENV_NAMES[@]}"; do
    for src in "${SRC_TYPES[@]}"; do
        for tar in "${TAR_TYPES[@]}"; do
            
            # 定义每个实验的日志文件名，方便排查
            LOG_FILE="${LOG_DIR}/${ENV_NAME}-src-${src}-tar-${tar}-${SEED}.log"
            
            echo "----------------------------------------------------"
            echo "Running: Env=${ENV_NAME}, Src=${src}, Tar=${tar}"
            echo "Params: Eps=${EPSILON}, L_src=${LAMBDA_SRC}, L_tar=${LAMBDA_TAR}, Thresh=${FILTER_THRESH}"
            echo "Log: ${LOG_FILE}"
            
            # 执行 Python 命令
            # nohup 不需要在这里加，我们在外部运行脚本时加
            # 使用 unbuffer (可选) 或 python -u 保证日志实时写入
            python -u run_ot.py \
                --env "${ENV_NAME}" \
                --srctype "${src}" \
                --tartype "${tar}" \
                --epsilon "${EPSILON}" \
                --lambda_src "${LAMBDA_SRC}" \
                --lambda_tar "${LAMBDA_TAR}" \
                --filter_threshold "${FILTER_THRESH}" \
                --metric "euclidean" \
                --seed "${SEED}" \
                --save-model \
                > "${LOG_FILE}" 2>&1
            
            # 检查上一个命令是否成功
            if [ $? -eq 0 ]; then
                echo "Success: ${ENV_NAME} ${src}->${tar}"
            else
                echo "Failed: ${ENV_NAME} ${src}->${tar} (Check ${LOG_FILE})"
            fi

        done
    done
done

echo "All experiments finished."
