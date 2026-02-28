"""
Parameterized SID generation from trained RQVAE checkpoint.

Usage:
    python generate_indices_replicate.py \
        --dataset Industrial_and_Scientific \
        --ckpt_path ./output_replication/Industrial_and_Scientific/<timestamp>/best_collision_model.pth \
        --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
        --output_dir ./output_replication/
"""

import argparse
import collections
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import EmbDataset
from models.rqvae import RQVAE


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Semantic IDs from RQVAE checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. Industrial_and_Scientific)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to best_collision_model.pth")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to .emb-qwen-td.npy (if not set, uses path from checkpoint args)")
    parser.add_argument("--output_dir", type=str, default="./output_replication/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--sk_epsilon_last", type=float, default=0.003,
                        help="Sinkhorn epsilon for last VQ layer during collision resolution")
    parser.add_argument("--max_sk_iters", type=int, default=20, help="Max sinkhorn collision resolution iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data_path = args.data_path if args.data_path else ckpt_args.data_path
    data = EmbDataset(data_path)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(data, num_workers=4, batch_size=64, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader, desc="Generating indices"):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # Set up sinkhorn for collision resolution on last layer
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = args.sk_epsilon_last

    tt = 0
    while True:
        if tt >= args.max_sk_iters or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(f"Iteration {tt}: {len(collision_item_groups)} collision groups")
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number:", len(all_indices))
    print("Max number of conflicts:", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate:", (tot_item - tot_indice) / tot_item)

    # Save
    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}.index.json")
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp)

    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
