# Position Reward Experiment Log

## 2026-02-27: Position Reward RL (Industrial_and_Scientific)

### Motivation

Existing reward types are either binary (rule: 0 or 1 for exact match) or ranking-based (rule + NDCG). Neither gives partial credit for getting some SID levels correct. The **position reward** sums consecutive correct SID positions from left, giving a smoother gradient signal:

- 0 positions correct → **0**
- 1st level correct → **1**
- 1st + 2nd correct → **2**
- All 3 correct → **3**

This is similar to the HEPO reward in `rl_gpr.py` (0.2 / 0.5 / 1.0) but uses uniform increments instead of hand-tuned values.

### Setup

- **Base model**: Qwen2.5-1.5B-Instruct (SFT checkpoint)
- **GPUs**: 8x A100-40GB (new instance)
- **Category**: Industrial_and_Scientific (3,686 items)
- **Evaluation**: 50-beam constrained decoding on test set

### RL Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| Per-device batch size | 64 | Same as 4-GPU runs |
| Gradient accumulation | **2** | Was 4 on 4 GPUs → 2 on 8 GPUs |
| Effective batch size | 1024 | 64 × 8 GPUs × 2 = 1024 (matched) |
| Learning rate | 1e-5 | Cosine with warmup |
| Num generations | 16 | |
| Beta (KL penalty) | 1e-3 | |
| Temperature | 1.0 | |
| **Reward type** | **position** | New: cumulative SID level score |
| Beam search | True | |
| Sync ref model | True | |
| Num epochs | 2 | |
| DeepSpeed | ZeRO-2 | |

- **Training**: 1,650 steps, ~3h12m
- **Final train loss**: 0.030
- **Reward trajectory**: Started ~0.43, fluctuated 0.3–0.6 throughout

### Training Dynamics

| Metric | Early (step ~5) | Mid (step ~420) | Final (step 1650) |
|--------|----------------|-----------------|-------------------|
| Position reward | 0.43 | 0.43–0.60 | 0.42–0.59 |
| KL | 0.0 | 10–22 | 1.4–2.4 |
| Loss | 0.0 | 0.01–0.02 | 0.002 |

Note: KL peaks mid-training then decreases, consistent with cosine LR schedule decay.

---

### Recommendation Metrics (Industrial_and_Scientific)

| Stage | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|-------|------|------|------|-------|-------|-------|--------|--------|--------|---------|---------|---------|
| SFT | 0.0620 | 0.0900 | 0.1059 | 0.1379 | 0.1719 | 0.2285 | 0.0620 | 0.0783 | 0.0848 | 0.0951 | 0.1037 | 0.1149 |
| RL (ranking, 2ep) | 0.0704 | 0.0966 | 0.1116 | 0.1396 | 0.1749 | 0.2303 | 0.0704 | 0.0855 | 0.0917 | 0.1007 | 0.1096 | 0.1205 |
| **RL (position, 2ep)** | **0.0613** | **0.0887** | **0.1044** | **0.1350** | **0.1632** | **0.2122** | **0.0613** | **0.0768** | **0.0833** | **0.0932** | **0.1004** | **0.1101** |

### Delta vs SFT

| Metric | Ranking reward | Position reward |
|--------|---------------|-----------------|
| HR@1 | **+0.0084** (+13.5%) | -0.0007 (-1.1%) |
| HR@3 | **+0.0066** (+7.3%) | -0.0013 (-1.4%) |
| HR@5 | **+0.0057** (+5.4%) | -0.0015 (-1.4%) |
| HR@10 | **+0.0017** (+1.2%) | -0.0029 (-2.1%) |
| HR@20 | **+0.0030** (+1.7%) | -0.0087 (-5.1%) |
| HR@50 | **+0.0018** (+0.8%) | -0.0163 (-7.1%) |
| NDCG@1 | **+0.0084** (+13.5%) | -0.0007 (-1.1%) |
| NDCG@5 | **+0.0069** (+8.1%) | -0.0015 (-1.8%) |
| NDCG@10 | **+0.0056** (+5.9%) | -0.0019 (-2.0%) |

**Position reward underperforms both SFT and ranking reward.** The model slightly regresses from SFT at all K values, with the gap widening at higher K.

---

### Per-Level Semantic ID Analysis

#### Independent Per-Level Hit Rate (top-K)

**SFT:**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2440 | 0.1233 | 0.0724 | 0.0620 |
| 5 | 0.3585 | 0.1944 | 0.1394 | 0.1059 |
| 10 | 0.4161 | 0.2358 | 0.1935 | 0.1379 |
| 50 | 0.5795 | 0.4072 | 0.3977 | 0.2285 |

**RL (ranking):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2645 | 0.1368 | 0.0814 | 0.0704 |
| 5 | 0.3552 | 0.1886 | 0.1513 | 0.1116 |
| 10 | 0.3905 | 0.2171 | 0.2001 | 0.1396 |
| 50 | 0.4814 | 0.3735 | 0.3997 | 0.2303 |

**RL (position):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2797 | 0.1370 | 0.0713 | 0.0613 |
| 5 | 0.3254 | 0.1670 | 0.1443 | 0.1043 |
| 10 | 0.3512 | 0.1904 | 0.1974 | 0.1350 |
| 50 | 0.4335 | 0.3194 | 0.3938 | 0.2122 |

#### Cumulative Level Accuracy (top-K)

**SFT:**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2440 | 0.1189 | 0.0620 |
| 5 | 0.3585 | 0.1756 | 0.1059 |
| 50 | 0.5795 | 0.3031 | 0.2285 |

**RL (ranking):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2645 | 0.1317 | 0.0704 |
| 5 | 0.3552 | 0.1719 | 0.1116 |
| 50 | 0.4814 | 0.2733 | 0.2303 |

**RL (position):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2797 | 0.1332 | 0.0613 |
| 5 | 0.3254 | 0.1562 | 0.1043 |
| 50 | 0.4335 | 0.2442 | 0.2122 |

#### Drop-off Analysis (top-1)

| Transition | SFT | Ranking | Position |
|------------|-----|---------|----------|
| L1 accuracy | 24.40% | 26.45% | **27.97%** |
| L1 → L1+L2 retention | 48.73% | 49.79% | **47.63%** |
| L1+L2 → L1+L2+L3 retention | 52.13% | 53.43% | **46.03%** |

---

### Key Observations

1. **Position reward improves L1 accuracy the most**: Top-1 L1 accuracy is 27.97% (vs 26.45% ranking, 24.40% SFT). The reward signal successfully encourages getting the first SID level correct.

2. **But retention through deeper levels is worse**: L1→L1+L2 retention drops to 47.63% (vs 49.79% ranking) and L1+L2→L1+L2+L3 retention drops sharply to 46.03% (vs 53.43% ranking). The model learns to get L1 right but at the cost of deeper accuracy.

3. **Exact match degrades below SFT**: HR@1 = 0.0613 (vs 0.0620 SFT, 0.0704 ranking). The position reward's partial credit at shallow levels does not translate to better final recommendations.

4. **Diversity decreases at higher K**: HR@50 drops to 0.2122 (vs 0.2285 SFT, 0.2303 ranking), suggesting the model concentrates its beam search on getting L1 correct but produces less diverse deeper predictions.

5. **Hypothesis for underperformance**: The uniform +1 per level reward may over-weight L1 correctness relative to its contribution to final recommendation quality. The ranking reward's binary exact-match signal with NDCG ranking more directly optimizes for the evaluation metric. The HEPO reward's non-uniform weights (0.2/0.5/1.0) may be a better balance.

---

## 2026-02-27: Position Reward RL (Office_Products, 4 epochs)

### Setup

- **Base model**: Qwen2.5-1.5B-Instruct (SFT checkpoint for Office)
- **GPUs**: 8x A100-40GB
- **Category**: Office_Products (38,924 train / 4,866 test)
- **Evaluation**: 50-beam constrained decoding on test set
- **Epochs**: **4** (doubled from Industrial's 2 to test longer training)

### RL Hyperparameters

Same as Industrial except:

| Parameter | Value | Note |
|-----------|-------|------|
| **Num epochs** | **4** | Doubled from Industrial (2ep) |
| All other params | Same | See Industrial section above |

- **Training**: 3,456 steps, ~5h52m
- **Effective batch size**: 1024 (64 × 8 GPUs × 2 grad_accum)

---

### Recommendation Metrics (Office_Products)

| Stage | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|-------|------|------|------|-------|-------|-------|--------|--------|--------|---------|---------|---------|
| SFT | 0.0812 | 0.1151 | 0.1317 | 0.1576 | 0.1860 | 0.2326 | 0.0812 | 0.1010 | 0.1079 | 0.1163 | 0.1234 | 0.1327 |
| RL (ranking, 2ep) | 0.0890 | 0.1215 | 0.1352 | 0.1562 | 0.1794 | 0.2201 | 0.0890 | 0.1078 | 0.1135 | 0.1203 | 0.1260 | 0.1340 |
| RL (position, ~2.4ep) | 0.0841 | 0.1141 | 0.1256 | 0.1449 | 0.1712 | 0.2100 | 0.0841 | 0.1016 | 0.1063 | 0.1125 | 0.1192 | 0.1268 |
| **RL (position, 4ep)** | **0.0810** | **0.1099** | **0.1293** | **0.1476** | **0.1693** | **0.2094** | **0.0810** | **0.0979** | **0.1057** | **0.1117** | **0.1171** | **0.1250** |

Note: "~2.4ep" is checkpoint-2074 (closest saved checkpoint to 2 epochs = step 1728; 2074/864 steps-per-epoch ≈ 2.4 epochs).

### Delta vs SFT (Office_Products)

| Metric | Ranking (2ep) | Position (~2.4ep) | Position (4ep) |
|--------|---------------|-------------------|----------------|
| HR@1 | **+0.0078** (+9.6%) | +0.0029 (+3.6%) | -0.0002 (-0.2%) |
| HR@3 | **+0.0064** (+5.6%) | -0.0010 (-0.9%) | -0.0052 (-4.5%) |
| HR@5 | **+0.0035** (+2.7%) | -0.0061 (-4.6%) | -0.0024 (-1.8%) |
| HR@10 | -0.0014 (-0.9%) | -0.0127 (-8.1%) | -0.0100 (-6.3%) |
| HR@20 | -0.0066 (-3.5%) | -0.0148 (-8.0%) | -0.0167 (-9.0%) |
| HR@50 | -0.0125 (-5.4%) | -0.0226 (-9.7%) | -0.0232 (-10.0%) |
| NDCG@1 | **+0.0078** (+9.6%) | +0.0029 (+3.6%) | -0.0002 (-0.2%) |
| NDCG@5 | **+0.0056** (+5.2%) | -0.0016 (-1.5%) | -0.0022 (-2.0%) |
| NDCG@10 | **+0.0040** (+3.4%) | -0.0038 (-3.3%) | -0.0046 (-4.0%) |

**Position reward underperforms both SFT and ranking reward at both checkpoints.** The ~2.4-epoch checkpoint is actually slightly better than the 4-epoch final model at HR@1 (+0.0029 vs -0.0002 relative to SFT), suggesting the model peaks early and then degrades with continued position reward training. The degradation at higher K is severe on both checkpoints (~8-10% below SFT at HR@50).

---

### Per-Level Semantic ID Analysis (Office_Products)

#### Independent Per-Level Hit Rate (top-K)

**SFT:**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1708 | 0.1017 | 0.0857 | 0.0812 |
| 5 | 0.2813 | 0.1693 | 0.1513 | 0.1317 |
| 10 | 0.3492 | 0.2139 | 0.1985 | 0.1576 |
| 50 | 0.5645 | 0.3820 | 0.4001 | 0.2326 |

**RL (ranking, 2ep):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1759 | 0.1095 | 0.0933 | 0.0890 |
| 5 | 0.2587 | 0.1661 | 0.1562 | 0.1352 |
| 10 | 0.3140 | 0.2028 | 0.1944 | 0.1562 |
| 50 | 0.4587 | 0.3615 | 0.3876 | 0.2201 |

**RL (position, ~2.4ep):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1913 | 0.1116 | 0.0882 | 0.0841 |
| 5 | 0.2394 | 0.1492 | 0.1486 | 0.1256 |
| 10 | 0.2762 | 0.1788 | 0.1889 | 0.1449 |
| 50 | 0.3594 | 0.3239 | 0.3876 | 0.2100 |

**RL (position, 4ep):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1913 | 0.1106 | 0.0847 | 0.0810 |
| 5 | 0.2460 | 0.1513 | 0.1519 | 0.1293 |
| 10 | 0.2867 | 0.1819 | 0.1905 | 0.1476 |
| 50 | 0.3658 | 0.3224 | 0.3903 | 0.2094 |

#### Cumulative Level Accuracy (top-K)

**SFT:**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1708 | 0.0986 | 0.0812 |
| 5 | 0.2813 | 0.1494 | 0.1317 |
| 50 | 0.5645 | 0.2635 | 0.2326 |

**RL (ranking, 2ep):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1759 | 0.1052 | 0.0890 |
| 5 | 0.2587 | 0.1490 | 0.1352 |
| 50 | 0.4587 | 0.2409 | 0.2201 |

**RL (position, ~2.4ep):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1913 | 0.1071 | 0.0841 |
| 5 | 0.2394 | 0.1350 | 0.1256 |
| 50 | 0.3594 | 0.2162 | 0.2100 |

**RL (position, 4ep):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1913 | 0.1058 | 0.0810 |
| 5 | 0.2460 | 0.1379 | 0.1293 |
| 50 | 0.3658 | 0.2156 | 0.2094 |

#### Drop-off Analysis (top-1, Office_Products)

| Transition | SFT | Ranking (2ep) | Position (~2.4ep) | Position (4ep) |
|------------|-----|---------------|-------------------|----------------|
| L1 accuracy | 17.08% | 17.59% | **19.13%** | **19.13%** |
| L1 → L1+L2 retention | 57.76% | 59.81% | 55.96% | 55.32% |
| L1+L2 → L1+L2+L3 retention | 82.29% | 84.57% | 78.50% | 76.50% |

#### Epoch Progression Analysis

L1 accuracy plateaus at 19.13% by ~2.4 epochs and stays identical at 4 epochs. However, deeper level retention **degrades** with continued training:
- L1→L1+L2: 55.96% → 55.32% (-0.6pp)
- L1+L2→L3: 78.50% → 76.50% (-2.0pp)

This confirms that additional position reward training beyond ~2 epochs is counterproductive — the model keeps optimizing L1 (already saturated) while eroding deeper-level precision.

---

## Cross-Dataset Summary

### Position Reward: Consistent Pattern Across Datasets

| Observation | Industrial (2ep) | Office (~2.4ep) | Office (4ep) |
|-------------|-----------------|-----------------|--------------|
| L1 accuracy (top-1) | **27.97%** (best) | **19.13%** (best) | **19.13%** (same) |
| L1→L1+L2 retention | 47.63% (worst) | 55.96% (worst) | 55.32% (worse) |
| L1+L2→L1+L2+L3 retention | 46.03% (worst) | 78.50% (worst) | 76.50% (worse) |
| HR@1 vs SFT | -0.0007 (-1.1%) | +0.0029 (+3.6%) | -0.0002 (-0.2%) |
| HR@50 vs SFT | -0.0163 (-7.1%) | -0.0226 (-9.7%) | -0.0232 (-10.0%) |

### Key Conclusions

1. **Position reward consistently improves L1 accuracy** across both datasets (Industrial: +3.6pp over SFT, Office: +2.1pp over SFT). The reward signal is effective at teaching the model to predict the coarsest SID level.

2. **But retention through deeper levels degrades on both datasets.** The model sacrifices L2 and L3 accuracy to improve L1. On Office, L1+L2→L1+L2+L3 retention drops from 82.29% (SFT) to 76.50% — a large regression on a dataset where deep-level retention is normally very high.

3. **Doubling epochs (4 vs 2) actively hurts.** Comparing the ~2.4-epoch checkpoint to the 4-epoch final model on Office: HR@1 drops from 0.0841 to 0.0810, and L1+L2→L3 retention degrades from 78.50% to 76.50%. The model peaks early and then over-optimizes for L1 at the expense of deeper levels. L1 accuracy itself saturates at 19.13% by ~2.4 epochs with no further gains.

4. **The uniform +1 per level reward is fundamentally misaligned with evaluation metrics.** HR@K and NDCG@K only reward exact matches. Partial credit for L1 correctness creates a reward gradient that pulls the model toward L1 accuracy at the expense of the full-SID precision that matters for recommendation quality.

5. **Ranking reward remains the best reward type** for both datasets. Its binary exact-match signal with NDCG ranking directly optimizes for what the evaluation measures.

### Next Steps

- [x] ~~Run position reward on Office_Products for cross-dataset comparison~~
- [ ] Try normalized position reward (0.33/0.67/1.0) to reduce L1 over-emphasis
- [ ] Try combining position reward with ranking reward (multi-reward)
- [ ] Compare with HEPO reward (0.2/0.5/1.0) from rl_gpr.py
- [ ] Try position reward with exponential weighting (e.g., 1/4/9 or 0.1/0.3/1.0) to emphasize deeper levels
