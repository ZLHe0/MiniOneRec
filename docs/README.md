# GenRec Lab — Experiment Documentation

## Folder Structure (MMDD)

### 0225/
- **experiment_log.md** — Core replication results for SFT and RL (ranking reward) on Industrial_and_Scientific. Includes paper comparison, discrepancy analysis, per-level SID decoding analysis.
- **reported_result_from_minionerec.png** — Screenshot of paper-reported results (Table 1) for reference.

### 0226/
- **experiment_log.md** — Office_Products replication added. Cross-dataset comparison between Industrial and Office.

### 0227/
- **training_data_composition.md** — Documents SFT and RL training data composition: SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, and RL data pipeline.
- **reward_ablation.md** — Rule-only vs ranking reward ablation on Office_Products and Industrial_and_Scientific. Isolates the contribution of binary exact-match vs NDCG ranking signal.
- **position_reward_experiment.md** — New "position reward" (cumulative SID level score 0/1/2/3) tested on Industrial (2ep) and Office (2.4ep, 4ep). Key finding: improves L1 accuracy but degrades deeper-level retention and overall HR/NDCG.
