# Experiment Log

## 2025-02-25: Initial Replication Run (Industrial_and_Scientific)

### Setup
- **Base model**: Qwen2.5-1.5B-Instruct
- **GPUs**: 4x A40 (indices 3,4,5,6)
- **Category**: Industrial_and_Scientific (3,686 items)
- **Semantic IDs**: Pre-generated 3-level SIDs via RQ-Kmeans
- **Evaluation**: 50-beam constrained decoding on test set, CC=0 (all predictions are valid SIDs)

### SFT

| Parameter | Value |
|-----------|-------|
| Global batch size | 1024 |
| Micro batch size | 16 |
| Gradient accumulation | 16 (1024 / 16 / 4 GPUs) |
| Learning rate | 3e-4 (cosine with warmup) |
| Max epochs | 10 (early stopping patience=3) |
| Cutoff length | 512 |

- **Training**: Early stopped at epoch 6.5 (best eval_loss=1.525 at epoch 5.0)
- **Duration**: ~50 minutes

### RL (GRPO, 2 epochs)

| Parameter | Value |
|-----------|-------|
| Per-device batch size | 64 |
| Gradient accumulation | 4 |
| Effective batch size | 1024 (64 × 4 GPUs × 4 grad_accum) |
| Learning rate | 1e-5 (cosine with warmup) |
| Num generations | 16 |
| Beta (KL penalty) | 1e-3 |
| Temperature | 1.0 |
| Reward type | ranking (rule_reward + ndcg_rule_reward) |
| Beam search | True |
| Sync ref model | True |
| Num epochs | 2 |
| DeepSpeed | ZeRO-2 |

- **Training**: 1,650 steps, final loss ~0.01
- **Duration**: ~9.5 hours

### RL (GRPO, 3rd epoch — stress test)

Continued from checkpoint-1650 with `num_train_epochs=3` to see how much an additional epoch can push performance.

| Parameter | Change |
|-----------|--------|
| Num epochs | 3 (resumed from checkpoint-1650 at epoch 2) |
| All other params | Same as above |

- **Status**: In progress (~step 1823/2475 as of last check)
- **Results**: Pending

---

### Full Results

#### Our Replication (Industrial_and_Scientific, 50-beam)

| Stage | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|-------|------|------|------|-------|-------|-------|--------|--------|--------|---------|---------|---------|
| SFT | 0.0620 | 0.0900 | 0.1059 | 0.1379 | 0.1719 | 0.2285 | 0.0620 | 0.0783 | 0.0848 | 0.0951 | 0.1037 | 0.1149 |
| RL (2 epochs) | 0.0704 | 0.0966 | 0.1116 | 0.1396 | 0.1749 | 0.2303 | 0.0704 | 0.0855 | 0.0917 | 0.1007 | 0.1096 | 0.1205 |
| RL (3 epochs) | — | — | — | — | — | — | — | — | — | — | — | — |

#### Paper-Reported Scores (MiniOneRec, Industrial_and_Scientific)

From Table 1 (fig_1.png) in the paper:

| Method | HR@3 | HR@5 | HR@10 | NDCG@3 | NDCG@5 | NDCG@10 |
|--------|------|------|-------|--------|--------|---------|
| GRU4Rec | 0.0638 | 0.0774 | 0.0999 | 0.0542 | 0.0598 | 0.0669 |
| Caser | 0.0618 | 0.0717 | 0.0942 | 0.0514 | 0.0555 | 0.0628 |
| SASRec | 0.0790 | 0.0909 | 0.1088 | 0.0700 | 0.0748 | 0.0806 |
| HSTU | 0.0927 | 0.1037 | 0.1163 | 0.0885 | 0.0918 | 0.0958 |
| TIGER | 0.0852 | 0.1010 | 0.1321 | 0.0742 | 0.0807 | 0.0908 |
| LCRec | 0.0915 | 0.1057 | 0.1332 | 0.0805 | 0.0862 | 0.0952 |
| BIGRec | 0.0931 | 0.1092 | 0.1370 | 0.0841 | 0.0907 | 0.0997 |
| D3 | 0.1024 | 0.1213 | 0.1500 | 0.0991 | 0.0989 | 0.1082 |
| S-DPO | 0.1032 | 0.1238 | 0.1524 | 0.0906 | 0.0991 | 0.1082 |
| **MiniOneRec** | **0.1143** | **0.1321** | **0.1586** | **0.1011** | **0.1084** | **0.1167** |

---

### Discrepancy Analysis

Our replication scores are notably lower than the paper-reported MiniOneRec results:

| Metric | Paper | Ours (RL 2ep) | Gap | Relative |
|--------|-------|---------------|-----|----------|
| HR@3 | 0.1143 | 0.0966 | -0.0177 | -15.5% |
| HR@5 | 0.1321 | 0.1116 | -0.0205 | -15.5% |
| HR@10 | 0.1586 | 0.1396 | -0.0190 | -12.0% |
| NDCG@3 | 0.1011 | 0.0855 | -0.0156 | -15.4% |
| NDCG@5 | 0.1084 | 0.0917 | -0.0167 | -15.4% |
| NDCG@10 | 0.1167 | 0.1007 | -0.0160 | -13.7% |

Despite the gap, the **relative improvement pattern holds**: RL consistently improves over SFT, matching the paper's finding that GRPO with ranking rewards benefits generative recommendation.

---

### Per-Level Semantic ID Decoding Analysis

Semantic IDs have 3 hierarchical levels: `<a_X><b_Y><c_Z>`. This analysis measures accuracy at each level to understand where decoding errors occur.

Script: `calc_level.py`

#### Independent Per-Level Hit Rate (top-K)

Whether *any* prediction in top-K matches the target at each level independently.

**SFT:**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2440 | 0.1233 | 0.0724 | 0.0620 |
| 3 | 0.3203 | 0.1701 | 0.1134 | 0.0900 |
| 5 | 0.3585 | 0.1944 | 0.1394 | 0.1059 |
| 10 | 0.4161 | 0.2358 | 0.1935 | 0.1379 |
| 20 | 0.4756 | 0.2897 | 0.2608 | 0.1719 |
| 50 | 0.5795 | 0.4072 | 0.3977 | 0.2285 |

**RL (2 epochs):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2645 | 0.1368 | 0.0814 | 0.0704 |
| 3 | 0.3241 | 0.1690 | 0.1218 | 0.0966 |
| 5 | 0.3552 | 0.1886 | 0.1513 | 0.1116 |
| 10 | 0.3905 | 0.2171 | 0.2001 | 0.1396 |
| 20 | 0.4337 | 0.2709 | 0.2698 | 0.1749 |
| 50 | 0.4814 | 0.3735 | 0.3997 | 0.2303 |

#### Cumulative Level Accuracy (top-K)

Whether *any single prediction* in top-K matches L1, then L1+L2 together, then all 3 levels (strictest).

**SFT:**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2440 | 0.1189 | 0.0620 |
| 3 | 0.3203 | 0.1599 | 0.0900 |
| 5 | 0.3585 | 0.1756 | 0.1059 |
| 10 | 0.4161 | 0.2056 | 0.1379 |
| 20 | 0.4756 | 0.2394 | 0.1719 |
| 50 | 0.5795 | 0.3031 | 0.2285 |

**RL (2 epochs):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2645 | 0.1317 | 0.0704 |
| 3 | 0.3241 | 0.1595 | 0.0966 |
| 5 | 0.3552 | 0.1719 | 0.1116 |
| 10 | 0.3905 | 0.1915 | 0.1396 |
| 20 | 0.4337 | 0.2206 | 0.1749 |
| 50 | 0.4814 | 0.2733 | 0.2303 |

#### Drop-off Analysis (top-1)

| Transition | SFT | RL (2ep) |
|------------|-----|----------|
| L1 accuracy | 24.40% | 26.45% |
| L1 → L1+L2 retention | 48.73% | 49.79% |
| L1+L2 → L1+L2+L3 retention | 52.13% | 53.43% |

#### Key Observations

1. **Level 1 is the bottleneck**: Only ~25% of top-1 predictions get the coarsest level right. Each subsequent level roughly halves the accuracy.
2. **RL improves all levels proportionally**: RL gains ~2pp at L1, ~1.3pp at L1+L2, ~0.8pp at exact match — the improvement cascades down.
3. **Retention rates are similar**: Both SFT and RL lose ~50% of correct predictions at each level transition, suggesting the hierarchical difficulty is intrinsic to the SID structure rather than model-specific.
4. **Top-K helps significantly**: At top-50, L1 accuracy reaches ~48-58%, but exact match is still only ~23% — many candidates get the category right but miss finer details.
