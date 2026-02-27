#!/bin/bash
# Reward ablation: runs rule-only RL for both datasets sequentially
# Companion to existing ranking reward results in:
#   ./output/rl_office_qwen2.5-1.5b-instruct (ranking, Office)
#   ./output/rl_qwen2.5-1.5b-instruct (ranking, Industrial)
#
# This script produces:
#   ./output/rl_rule_office_qwen2.5-1.5b-instruct (rule, Office)
#   ./output/rl_rule_industrial_qwen2.5-1.5b-instruct (rule, Industrial)
#
# Usage: bash run_rl_reward_ablation.sh

set -e  # exit on error so a failure doesn't silently start the next run

echo "============================================"
echo "  Reward Ablation: Rule-Only RL Training"
echo "============================================"
echo ""

# --- Phase 1: Office_Products ---
echo "[Phase 1/2] Starting rule-only RL for Office_Products..."
echo "Start time: $(date)"
bash run_rl_rule_office.sh
echo "[Phase 1/2] Office_Products rule-only RL finished at $(date)"
echo ""

# --- Phase 2: Industrial_and_Scientific ---
echo "[Phase 2/2] Starting rule-only RL for Industrial_and_Scientific..."
echo "Start time: $(date)"
bash run_rl_rule_industrial.sh
echo "[Phase 2/2] Industrial_and_Scientific rule-only RL finished at $(date)"
echo ""

echo "============================================"
echo "  Reward Ablation Complete!"
echo "  Rule-only models saved to:"
echo "    ./output/rl_rule_office_qwen2.5-1.5b-instruct"
echo "    ./output/rl_rule_industrial_qwen2.5-1.5b-instruct"
echo ""
echo "  To evaluate, run:"
echo "    bash run_eval_office.sh ./output/rl_rule_office_qwen2.5-1.5b-instruct"
echo "    bash run_eval_industrial.sh ./output/rl_rule_industrial_qwen2.5-1.5b-instruct"
echo "============================================"
