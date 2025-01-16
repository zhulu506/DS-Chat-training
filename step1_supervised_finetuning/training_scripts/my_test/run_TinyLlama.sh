#!/bin/bash

OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

deepspeed --num_gpus 1 main.py \
   --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 32 \
   --learning_rate 1e-4 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --lora_dim 32 \
   --deepspeed \
   --output_dir $OUTPUT_PATH \