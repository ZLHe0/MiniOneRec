#!/bin/bash
# RL script for Industrial_and_Scientific with POSITION reward, configured for 8 GPUs
# Position reward: counts consecutive correct SID positions from left (0/1/2/3)
# Effective batch size: 64 * 8 GPUs * 2 grad_accum = 1024 (matches 4-GPU setup)

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/work/envs/MiniOneRec
export WANDB_MODE=offline

MODEL_PATH="./output/sft_qwen2.5-1.5b-instruct"
OUTPUT_DIR="./output/rl_position_industrial_qwen2.5-1.5b-instruct"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "Train: ${train_file}"
    echo "Eval:  ${eval_file}"
    echo "Info:  ${info_file}"

    accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 8 --main_process_port 29503 \
        rl.py \
        --model_path ${MODEL_PATH} \
        --train_batch_size 64 \
        --eval_batch_size 128 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type position \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir ${OUTPUT_DIR} \
        --wandb_run_name rl_position_industrial \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
