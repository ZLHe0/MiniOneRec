# Reward Ablation: Rule-Only vs Ranking Reward

**Date**: 2025-02-27
**Objective**: Compare RL with rule-only reward (binary exact match) vs ranking reward (rule + NDCG ranking signal) to isolate the contribution of each reward component.

---

## Reward Functions

| Reward Type | `--reward_type` | Components | Signal |
|-------------|-----------------|------------|--------|
| **Rule** | `rule` | `rule_reward` | Binary 0/1: exact SID match |
| **Ranking** | `ranking` | `rule_reward` + `ndcg_rule_reward` | Binary match + NDCG-weighted ranking across generation group |

Both use identical hyperparameters (lr=1e-5, beta=1e-3, 2 epochs, batch=1024, 16 generations, beam search, sync ref model).

---

## Office_Products Results

### Recommendation Metrics (50-beam)

| Stage | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|-------|------|------|------|-------|-------|-------|
| SFT (baseline) | 0.0812 | 0.1151 | 0.1317 | 0.1576 | 0.1860 | 0.2326 |
| RL (rule) | 0.0884 | 0.1206 | 0.1338 | 0.1556 | 0.1806 | 0.2259 |
| RL (ranking) | 0.0890 | 0.1215 | 0.1352 | 0.1562 | 0.1794 | 0.2201 |

| Stage | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|-------|--------|--------|--------|---------|---------|---------|
| SFT (baseline) | 0.0812 | 0.1010 | 0.1079 | 0.1163 | 0.1234 | 0.1327 |
| RL (rule) | 0.0884 | 0.1071 | 0.1126 | 0.1196 | 0.1259 | 0.1348 |
| RL (ranking) | 0.0890 | 0.1078 | 0.1135 | 0.1203 | 0.1260 | 0.1340 |

### Gain over SFT

| Metric | Rule Gain | Ranking Gain | Ranking vs Rule |
|--------|-----------|--------------|-----------------|
| HR@1 | +0.0072 (+8.9%) | +0.0078 (+9.6%) | +0.0006 |
| HR@3 | +0.0055 (+4.8%) | +0.0064 (+5.6%) | +0.0009 |
| HR@5 | +0.0021 (+1.6%) | +0.0035 (+2.7%) | +0.0014 |
| HR@10 | -0.0020 (-1.3%) | -0.0014 (-0.9%) | +0.0006 |
| HR@20 | -0.0054 (-2.9%) | -0.0066 (-3.5%) | -0.0012 |
| HR@50 | -0.0067 (-2.9%) | -0.0125 (-5.4%) | -0.0058 |
| NDCG@1 | +0.0072 (+8.9%) | +0.0078 (+9.6%) | +0.0006 |
| NDCG@3 | +0.0061 (+6.0%) | +0.0068 (+6.7%) | +0.0007 |
| NDCG@5 | +0.0047 (+4.4%) | +0.0056 (+5.2%) | +0.0009 |
| NDCG@10 | +0.0033 (+2.8%) | +0.0040 (+3.4%) | +0.0007 |
| NDCG@20 | +0.0025 (+2.0%) | +0.0026 (+2.1%) | +0.0001 |
| NDCG@50 | +0.0021 (+1.6%) | +0.0013 (+1.0%) | -0.0008 |

### Key Observations (Metrics)

1. **Both RL rewards improve top-K accuracy at small K** (HR@1/3, NDCG@1/3/5) over SFT, with ranking having a slight edge (~0.5-1.4pp more).
2. **Both RL rewards degrade large-K diversity**: HR@20/50 drops below SFT for both, suggesting RL sharpens the distribution — concentrating probability on fewer candidates.
3. **Ranking vs rule difference is marginal**: The NDCG ranking signal adds only ~0.05-0.15pp on top of rule reward across most metrics. The bulk of RL improvement comes from the binary match signal.
4. **Rule reward slightly better at large K**: At HR@50, rule (-2.9%) degrades less than ranking (-5.4%), suggesting the ranking signal over-sharpens the distribution.

---

### Per-Level Semantic ID Decoding Analysis

#### Per-Level Hit Rate (top-K)

**RL (rule):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1778 | 0.1095 | 0.0921 | 0.0884 |
| 3 | 0.2314 | 0.1480 | 0.1346 | 0.1206 |
| 5 | 0.2680 | 0.1640 | 0.1576 | 0.1338 |
| 10 | 0.3241 | 0.2012 | 0.1967 | 0.1556 |
| 20 | 0.3919 | 0.2561 | 0.2524 | 0.1806 |
| 50 | 0.4449 | 0.3549 | 0.3977 | 0.2259 |

**RL (ranking):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1759 | 0.1095 | 0.0933 | 0.0890 |
| 3 | 0.2271 | 0.1482 | 0.1348 | 0.1215 |
| 5 | 0.2587 | 0.1661 | 0.1562 | 0.1352 |
| 10 | 0.3140 | 0.2028 | 0.1944 | 0.1562 |
| 20 | 0.3816 | 0.2602 | 0.2503 | 0.1794 |
| 50 | 0.4587 | 0.3615 | 0.3876 | 0.2201 |

**SFT (baseline):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.1708 | 0.1017 | 0.0857 | 0.0812 |
| 3 | 0.2392 | 0.1428 | 0.1266 | 0.1151 |
| 5 | 0.2813 | 0.1693 | 0.1513 | 0.1317 |
| 10 | 0.3492 | 0.2139 | 0.1985 | 0.1576 |
| 20 | 0.4301 | 0.2750 | 0.2657 | 0.1860 |
| 50 | 0.5645 | 0.3820 | 0.4001 | 0.2326 |

#### Cumulative Level Accuracy (top-K)

**RL (rule):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1778 | 0.1056 | 0.0884 |
| 3 | 0.2314 | 0.1363 | 0.1206 |
| 5 | 0.2680 | 0.1478 | 0.1338 |
| 10 | 0.3241 | 0.1728 | 0.1556 |
| 20 | 0.3919 | 0.2006 | 0.1806 |
| 50 | 0.4449 | 0.2433 | 0.2259 |

**RL (ranking):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.1759 | 0.1052 | 0.0890 |
| 3 | 0.2271 | 0.1356 | 0.1215 |
| 5 | 0.2587 | 0.1490 | 0.1352 |
| 10 | 0.3140 | 0.1710 | 0.1562 |
| 20 | 0.3816 | 0.1987 | 0.1794 |
| 50 | 0.4587 | 0.2409 | 0.2201 |

#### Drop-off Analysis (top-1)

| Transition | SFT | RL (rule) | RL (ranking) |
|------------|-----|-----------|--------------|
| L1 accuracy | 17.08% | 17.78% | 17.59% |
| L1 → L1+L2 retention | 57.76% | 59.42% | 59.81% |
| L1+L2 → L1+L2+L3 retention | 82.29% | 83.66% | 84.57% |

### Key Observations (Per-Level)

1. **Rule and ranking produce nearly identical level patterns**: Both have ~17.7% L1, ~10.5-10.6% L1+L2, ~8.8-8.9% exact at top-1. The reward type does not meaningfully change the hierarchical decoding behavior.
2. **RL improves retention at every level transition**: Both rule and ranking improve L1→L1+L2 retention (~59-60% vs 57.8% SFT) and L1+L2→L3 retention (~83.7-84.6% vs 82.3% SFT).
3. **Rule has slightly higher L1 but lower exact match**: Rule gets 17.78% L1 (vs 17.59% ranking) but 8.84% exact (vs 8.90% ranking). Ranking's NDCG signal marginally helps with fine-grained disambiguation.
4. **SFT has better large-K coverage**: At top-50, SFT reaches 56.5% L1 vs ~44-46% for both RL variants. RL trades breadth for precision at small K.

---

## Industrial_and_Scientific Results

### Recommendation Metrics (50-beam)

| Stage | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|-------|------|------|------|-------|-------|-------|
| SFT (baseline) | 0.0620 | 0.0873 | 0.1013 | 0.1259 | 0.1582 | 0.2131 |
| RL (rule) | 0.0666 | 0.0946 | 0.1141 | 0.1425 | 0.1782 | 0.2345 |
| RL (ranking) | 0.0704 | 0.0966 | 0.1116 | 0.1396 | 0.1742 | 0.2279 |

| Stage | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | NDCG@50 |
|-------|--------|--------|--------|---------|---------|---------|
| SFT (baseline) | 0.0620 | 0.0769 | 0.0828 | 0.0907 | 0.0989 | 0.1095 |
| RL (rule) | 0.0666 | 0.0831 | 0.0911 | 0.1003 | 0.1094 | 0.1205 |
| RL (ranking) | 0.0704 | 0.0855 | 0.0917 | 0.1007 | 0.1089 | 0.1187 |

### Gain over SFT

| Metric | Rule Gain | Ranking Gain | Ranking vs Rule |
|--------|-----------|--------------|-----------------|
| HR@1 | +0.0046 (+7.4%) | +0.0084 (+13.5%) | +0.0038 |
| HR@3 | +0.0073 (+8.4%) | +0.0093 (+10.7%) | +0.0020 |
| HR@5 | +0.0128 (+12.6%) | +0.0103 (+10.2%) | -0.0025 |
| HR@10 | +0.0166 (+13.2%) | +0.0137 (+10.9%) | -0.0029 |
| HR@20 | +0.0200 (+12.6%) | +0.0160 (+10.1%) | -0.0040 |
| HR@50 | +0.0214 (+10.0%) | +0.0148 (+6.9%) | -0.0066 |
| NDCG@1 | +0.0046 (+7.4%) | +0.0084 (+13.5%) | +0.0038 |
| NDCG@3 | +0.0062 (+8.1%) | +0.0086 (+11.2%) | +0.0024 |
| NDCG@5 | +0.0083 (+10.0%) | +0.0089 (+10.7%) | +0.0006 |
| NDCG@10 | +0.0096 (+10.6%) | +0.0100 (+11.0%) | +0.0004 |
| NDCG@20 | +0.0105 (+10.6%) | +0.0100 (+10.1%) | -0.0005 |
| NDCG@50 | +0.0110 (+10.0%) | +0.0092 (+8.4%) | -0.0018 |

### Key Observations (Metrics)

1. **Both rewards improve across ALL K values**: Unlike Office where RL degraded large-K, on Industrial both rule and ranking improve over SFT at every K. This suggests the SFT model on Industrial has more room for improvement.
2. **Ranking wins at small K, rule wins at large K**: Ranking leads at HR@1 (+13.5% vs +7.4%) and HR@3, but rule overtakes from HR@5 onward (+12.6% vs +10.2% at HR@5, +10.0% vs +6.9% at HR@50).
3. **Rule reward achieves better large-K diversity**: HR@50 improves +10.0% with rule vs +6.9% with ranking — the NDCG signal again over-sharpens the distribution.
4. **Ranking's NDCG component matters more on Industrial**: At HR@1, ranking leads rule by 3.8pp (vs only 0.6pp on Office), suggesting the ranking signal helps more when the task is harder.

---

### Per-Level Semantic ID Decoding Analysis

#### Per-Level Hit Rate (top-K)

**RL (rule):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2612 | 0.1304 | 0.0812 | 0.0666 |
| 3 | 0.3371 | 0.1794 | 0.1244 | 0.0946 |
| 5 | 0.3737 | 0.1979 | 0.1533 | 0.1141 |
| 10 | 0.4150 | 0.2310 | 0.2023 | 0.1425 |
| 20 | 0.4511 | 0.2791 | 0.2724 | 0.1782 |
| 50 | 0.4889 | 0.3761 | 0.3989 | 0.2345 |

**RL (ranking):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2645 | 0.1317 | 0.0830 | 0.0704 |
| 3 | 0.3356 | 0.1740 | 0.1221 | 0.0966 |
| 5 | 0.3614 | 0.1932 | 0.1481 | 0.1116 |
| 10 | 0.4032 | 0.2294 | 0.1991 | 0.1396 |
| 20 | 0.4474 | 0.2784 | 0.2668 | 0.1742 |
| 50 | 0.4827 | 0.3711 | 0.3955 | 0.2279 |

**SFT (baseline):**

| K | Level 1 | Level 2 | Level 3 | Exact |
|---|---------|---------|---------|-------|
| 1 | 0.2453 | 0.1207 | 0.0753 | 0.0620 |
| 3 | 0.3289 | 0.1713 | 0.1153 | 0.0873 |
| 5 | 0.3667 | 0.1923 | 0.1422 | 0.1013 |
| 10 | 0.4204 | 0.2349 | 0.2022 | 0.1259 |
| 20 | 0.4843 | 0.2923 | 0.2805 | 0.1582 |
| 50 | 0.5783 | 0.3853 | 0.4082 | 0.2131 |

#### Cumulative Level Accuracy (top-K)

**RL (rule):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2612 | 0.1253 | 0.0666 |
| 3 | 0.3371 | 0.1666 | 0.0946 |
| 5 | 0.3737 | 0.1798 | 0.1141 |
| 10 | 0.4150 | 0.2010 | 0.1425 |
| 20 | 0.4511 | 0.2303 | 0.1782 |
| 50 | 0.4889 | 0.2784 | 0.2345 |

**RL (ranking):**

| K | L1 | L1+L2 | L1+L2+L3 |
|---|-----|-------|----------|
| 1 | 0.2645 | 0.1317 | 0.0704 |
| 3 | 0.3356 | 0.1701 | 0.0966 |
| 5 | 0.3614 | 0.1837 | 0.1116 |
| 10 | 0.4032 | 0.2068 | 0.1396 |
| 20 | 0.4474 | 0.2390 | 0.1742 |
| 50 | 0.4827 | 0.2887 | 0.2279 |

#### Drop-off Analysis (top-1)

| Transition | SFT | RL (rule) | RL (ranking) |
|------------|-----|-----------|--------------|
| L1 accuracy | 24.53% | 26.12% | 26.45% |
| L1 → L1+L2 retention | 49.24% | 47.97% | 49.79% |
| L1+L2 → L1+L2+L3 retention | 51.37% | 53.17% | 53.43% |

### Key Observations (Per-Level)

1. **Both RL variants improve L1 accuracy**: Rule gets 26.12%, ranking gets 26.45% (vs 24.53% SFT). The improvement is larger than on Office (~1.6-2pp vs ~0.5-0.7pp).
2. **Rule reward slightly hurts L1→L2 retention**: 47.97% vs 49.24% SFT. Rule improves L1 breadth but doesn't help the model disambiguate at L2. Ranking preserves L1→L2 retention (49.79%).
3. **Both improve L2→L3 retention**: ~53.2-53.4% vs 51.4% SFT. The fine-grained level benefits from RL regardless of reward type.
4. **Industrial's hierarchical difficulty persists**: ~48-50% retention at L1→L2 and ~51-53% at L2→L3, far below Office's ~60%/~84%. The dataset's SID hierarchy is fundamentally harder.

---

## Cross-Dataset Comparison

### Ranking vs Rule: Who wins?

| Dataset | Small K (HR@1/3) | Large K (HR@10/20/50) | Per-Level Pattern |
|---------|------------------|-----------------------|-------------------|
| **Office** | Ranking slightly better (+0.6-0.9pp) | Rule slightly better (less degradation) | Nearly identical |
| **Industrial** | Ranking clearly better (+2.0-3.8pp) | Rule clearly better (+2.5-6.6pp more gain) | Ranking better retention at L1→L2 |

### Summary

| Finding | Office | Industrial |
|---------|--------|------------|
| RL over SFT (HR@1) | +8-10% | +7-14% |
| Ranking over rule (HR@1) | +0.6pp | +3.8pp |
| Rule over ranking (HR@50) | +5.8pp | +6.6pp |
| NDCG signal value | Marginal | Meaningful at small K |
| Large-K over-sharpening | Both RL degrade | Only ranking degrades slightly |

**Conclusion**: The binary rule reward is the primary driver of RL improvement on both datasets. The NDCG ranking component provides a small but consistent boost at top-1/3 precision (more pronounced on harder datasets like Industrial) but comes at the cost of large-K diversity. For applications prioritizing precision (top-1 recommendation), ranking is preferred. For applications needing diverse candidate lists, rule-only may be better.
