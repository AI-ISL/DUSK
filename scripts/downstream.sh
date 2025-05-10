#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --output=logs/downstream_eval_%j.out
#SBATCH --error=logs/downstream_eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

export HF_HUB_READ_TIMEOUT=60

# log directory
mkdir -p logs

# put your model paths here
MODEL_PATHS=("results/llama3-8b/GA/seed_1001/epoch1_1e-05_taskformats_5/1/unlearn_times_1/checkpoint-last" )

TASKS=("arc_challenge" "truthfulqa" "triviaqa" "mmlu" "gsm8k" "bigbench_bbq_lite_json_multiple_choice")

OUTPUT_DIR="./downstream_results"

srun python downstream.py \
  --model_paths $MODEL_PATHS \
  --tasks $TASKS \
  --output_dir $OUTPUT_DIR