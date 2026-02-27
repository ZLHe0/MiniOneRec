#!/bin/bash
# RL script for Office_Products, configured for 4 GPUs (3,4,5,6)

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=3,4,5,6

MODEL_PATH="./output/sft_office_qwen2.5-1.5b-instruct"
OUTPUT_DIR="./output/rl_office_qwen2.5-1.5b-instruct"

for category in "Office_Products"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "Train: ${train_file}"
    echo "Eval:  ${eval_file}"
    echo "Info:  ${info_file}"

    accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
        rl.py \
        --model_path ${MODEL_PATH} \
        --train_batch_size 64 \
        --eval_batch_size 128 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 4 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
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
        --wandb_run_name rl_office \
        --sid_index_path ./data/Amazon/index/Office_Products.index.json \
        --item_meta_path ./data/Amazon/index/Office_Products.item.json
done
