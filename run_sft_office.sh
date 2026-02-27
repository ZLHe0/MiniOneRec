#!/bin/bash
# SFT script for Office_Products, configured for 4 GPUs (3,4,5,6)

export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=3,4,5,6

BASE_MODEL="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUTPUT_DIR="./output/sft_office_qwen2.5-1.5b-instruct"

for category in "Office_Products"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo "Train: ${train_file}"
    echo "Eval:  ${eval_file}"
    echo "Info:  ${info_file}"

    torchrun --nproc_per_node 4 \
            sft.py \
            --base_model ${BASE_MODEL} \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ${OUTPUT_DIR} \
            --wandb_project minionerec_repro \
            --wandb_run_name sft_office \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Office_Products.index.json \
            --item_meta_path ./data/Amazon/index/Office_Products.item.json \
            --freeze_LLM False
done
