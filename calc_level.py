"""
Per-level semantic ID accuracy analysis.

Semantic IDs have 3 levels: <a_X><b_Y><c_Z>
This script computes how often the model gets each level correct,
both for the top-1 prediction and across top-K predictions.

Usage:
    python calc_level.py --path results/rl_qwen2.5-1.5b-instruct/final_result_Industrial_and_Scientific.json
"""

import json
import re
import fire
import numpy as np
from collections import defaultdict


def parse_sid(sid):
    """Parse a semantic ID string into its 3 levels.

    E.g., '<a_223><b_80><c_216>' -> ['<a_223>', '<b_80>', '<c_216>']
    Returns None if parsing fails.
    """
    sid = sid.strip(' "\n')
    parts = re.findall(r'<[abc]_\d+>', sid)
    if len(parts) == 3:
        return parts
    return None


def calc_level(path, topk_list=[1, 3, 5, 10, 20, 50]):
    with open(path, 'r') as f:
        data = json.load(f)

    n_samples = len(data)
    n_beams = len(data[0]['predict'])
    valid_topk = [k for k in topk_list if k <= n_beams]

    # Per-level accuracy at each top-K
    # level_hits[k][level] = count of samples where level is correct within top-k
    level_hits = {k: [0, 0, 0] for k in valid_topk}
    # All-level (exact match) hits at each top-K
    exact_hits = {k: 0 for k in valid_topk}
    # Per-level accuracy for top-1 only (detailed breakdown)
    top1_level = [0, 0, 0]

    parse_failures = 0

    for sample in data:
        target = sample['output'].strip(' "\n')
        target_parts = parse_sid(target)
        if target_parts is None:
            parse_failures += 1
            continue

        predictions = [p.strip(' "\n') for p in sample['predict']]

        # For each top-K, check if any prediction within top-K matches at each level
        for k in valid_topk:
            top_k_preds = predictions[:k]
            level_matched = [False, False, False]
            exact_matched = False

            for pred in top_k_preds:
                pred_parts = parse_sid(pred)
                if pred_parts is None:
                    continue

                if pred_parts == target_parts:
                    exact_matched = True
                    level_matched = [True, True, True]
                    break

                for lvl in range(3):
                    if pred_parts[lvl] == target_parts[lvl]:
                        level_matched[lvl] = True

            for lvl in range(3):
                if level_matched[lvl]:
                    level_hits[k][lvl] += 1
            if exact_matched:
                exact_hits[k] += 1

        # Top-1 detailed analysis
        if predictions:
            pred_parts = parse_sid(predictions[0])
            if pred_parts is not None:
                for lvl in range(3):
                    if pred_parts[lvl] == target_parts[lvl]:
                        top1_level[lvl] += 1

    valid_samples = n_samples - parse_failures
    if parse_failures > 0:
        print(f"Warning: {parse_failures}/{n_samples} samples failed to parse")

    # Print top-1 per-level breakdown
    print(f"\n{'='*60}")
    print(f"Per-Level Decoding Accuracy ({valid_samples} samples, {n_beams} beams)")
    print(f"{'='*60}")

    print(f"\n--- Top-1 Per-Level Accuracy ---")
    level_names = ['Level 1 (<a_X>)', 'Level 2 (<b_Y>)', 'Level 3 (<c_Z>)']
    for lvl in range(3):
        acc = top1_level[lvl] / valid_samples
        print(f"  {level_names[lvl]:25s}: {top1_level[lvl]:5d}/{valid_samples}  = {acc:.4f}")
    exact_top1 = exact_hits[1] if 1 in exact_hits else 0
    print(f"  {'Exact match (all 3)':25s}: {exact_top1:5d}/{valid_samples}  = {exact_top1/valid_samples:.4f}")

    # Print top-K table
    print(f"\n--- Top-K Per-Level Hit Rate ---")
    header = f"{'K':>5s} | {'Level1':>8s} | {'Level2':>8s} | {'Level3':>8s} | {'Exact':>8s}"
    print(header)
    print("-" * len(header))
    for k in valid_topk:
        l1 = level_hits[k][0] / valid_samples
        l2 = level_hits[k][1] / valid_samples
        l3 = level_hits[k][2] / valid_samples
        ex = exact_hits[k] / valid_samples
        print(f"{k:5d} | {l1:8.4f} | {l2:8.4f} | {l3:8.4f} | {ex:8.4f}")

    # Cumulative level analysis at each top-K
    # For each K: among top-K predictions, is there one that matches L1? L1+L2? L1+L2+L3?
    print(f"\n--- Top-K Cumulative Level Accuracy ---")
    print(f"  (L1 correct, then L1+L2 both correct, then all 3 correct)")

    cum_topk = {k: [0, 0, 0] for k in valid_topk}
    for sample in data:
        target_parts = parse_sid(sample['output'])
        if target_parts is None:
            continue
        predictions = [p.strip(' "\n') for p in sample['predict']]

        for k in valid_topk:
            best_cum = 0  # track deepest cumulative match in top-k
            for pred in predictions[:k]:
                pred_parts = parse_sid(pred)
                if pred_parts is None:
                    continue
                if pred_parts[0] == target_parts[0]:
                    depth = 1
                    if pred_parts[1] == target_parts[1]:
                        depth = 2
                        if pred_parts[2] == target_parts[2]:
                            depth = 3
                    best_cum = max(best_cum, depth)
                    if best_cum == 3:
                        break
            for lvl in range(best_cum):
                cum_topk[k][lvl] += 1

    header = f"{'K':>5s} | {'L1':>8s} | {'L1+L2':>8s} | {'L1+L2+L3':>8s}"
    print(header)
    print("-" * len(header))
    for k in valid_topk:
        l1 = cum_topk[k][0] / valid_samples
        l12 = cum_topk[k][1] / valid_samples
        l123 = cum_topk[k][2] / valid_samples
        print(f"{k:5d} | {l1:8.4f} | {l12:8.4f} | {l123:8.4f}")

    # Drop-off analysis at top-1
    if cum_topk[1][0] > 0:
        print(f"\n--- Drop-off Analysis (top-1) ---")
        print(f"  L1 → L1+L2 retention: {cum_topk[1][1]/cum_topk[1][0]:.4f} ({cum_topk[1][0]-cum_topk[1][1]} lost)")
        if cum_topk[1][1] > 0:
            print(f"  L1+L2 → L1+L2+L3 retention: {cum_topk[1][2]/cum_topk[1][1]:.4f} ({cum_topk[1][1]-cum_topk[1][2]} lost)")


if __name__ == '__main__':
    fire.Fire(calc_level)
