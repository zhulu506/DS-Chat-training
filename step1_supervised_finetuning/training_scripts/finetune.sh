#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# shell config
GPU_NUM=$1
BS=$2
MODEL=$3
ZERO_STAGE=3
GMLAKE=$4 # True or False
if [ "$GMLAKE" == "True" ]; then
    export autoGC=10000
    export fragLimit=536870912
    export reuseLimit=10
    export defragLevel=0 # 0-greedy  1-lazy
    export GMLAKE_INFO=INFO
    export vmmDefragment=1
else
    unset autoGC
    unset fragLimit
    unset reuseLimit
    unset defragLevel
    unset GMLAKE_INFO
    export vmmDefragment=0
fi
L=$5
R=$6
O=$7

MODEL_NAME=$(echo ${MODEL} | cut -d / -f 2)
OUTPUT=./output/output_${MODEL_NAME}
mkdir -p $OUTPUT

LORA_NAME=""
if [ "$MODEL_NAME" == "opt-125m" ] || [ "$MODEL_NAME" == "opt-350m" ] || [ "$MODEL_NAME" == "opt-1.3b" ]; then
    LORA_NAME=decoder.layers.
fi
echo $LORA_NAME

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MAIN_ARGS=( --data_path  Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets 
   --data_split 2,4,4 
   --model_name_or_path ${MODEL} 
   --per_device_train_batch_size ${BS} 
   --per_device_eval_batch_size 1 
   --max_seq_len 512 
   --learning_rate 1e-3
   --weight_decay 0. 
   --num_train_epochs 16  
   --gradient_accumulation_steps 1 
   --lr_scheduler_type cosine 
   --num_warmup_steps 0 
   --seed 1234 
   --zero_stage $ZERO_STAGE 
   --deepspeed 
   --output_dir $OUTPUT 
)

if [ $R -eq 1 ]; then
    MAIN_ARGS+=( --gradient_checkpointing )
fi
if [ $L -eq 1 ]; then
    MAIN_ARGS+=( --lora_dim 128 --lora_module_name ${LORA_NAME} )
fi
if [ $O -eq 1 ]; then
    MAIN_ARGS+=( --offload )
fi

TIMESTAMP=$(date +"%Y%m%d%H%M%S")
if [ "$GMLAKE" == "True" ]; then
    LOG_FILE="${OUTPUT}/${MODEL_NAME}_B${BS}_GMLAKE-${GMLAKE}_L${L}_R${R}_O${O}_GC${autoGC}_FL${fragLimit}_RL${reuseLimit}_${TIMESTAMP}.log"
else
    LOG_FILE="${OUTPUT}/${MODEL_NAME}_B${BS}_GMLAKE-${GMLAKE}_L${L}_R${R}_O${O}_${TIMESTAMP}.log"
fi

deepspeed --num_gpus ${GPU_NUM} main.py \
   ${MAIN_ARGS[@]} \
   &> $LOG_FILE