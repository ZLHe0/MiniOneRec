#!/bin/bash
set -e

export WANDB_MODE=offline
export CUDA_HOME=/work/envs/MiniOneRec

cd "$(dirname "$0")"

echo "============================================="
echo "  RQVAE Replication - Industrial_and_Scientific"
echo "============================================="

python rqvae_wandb.py \
    --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir ./output_replication/Industrial_and_Scientific \
    --wandb_project rqvae-replication \
    --wandb_run_name rqvae_industrial \
    --lr 1e-3 --epochs 10000 --batch_size 20480 \
    --num_emb_list 256 256 256 --e_dim 32 \
    --layers 2048 1024 512 256 128 64 \
    --eval_step 50 --device cuda:0

echo "============================================="
echo "  RQVAE Replication - Office_Products"
echo "============================================="

python rqvae_wandb.py \
    --data_path ../data/Amazon/index/Office_Products.emb-qwen-td.npy \
    --ckpt_dir ./output_replication/Office_Products \
    --wandb_project rqvae-replication \
    --wandb_run_name rqvae_office \
    --lr 1e-3 --epochs 10000 --batch_size 20480 \
    --num_emb_list 256 256 256 --e_dim 32 \
    --layers 2048 1024 512 256 128 64 \
    --eval_step 50 --device cuda:0

echo "============================================="
echo "  Done! Check wandb offline runs in ./wandb/"
echo "============================================="
