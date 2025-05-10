#!/bin/bash
#SBATCH --job-name=DUSK
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/Unlearn_target_%J.out
#SBATCH --error=logs/Unlearn_target_%J.err

MASTER_PORT=$((RANDOM % 50001 + 10000))

forget_losses=(
    TV
)

task_list=(1)
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

default_epochss=(1 2 3 4 5)

lr=1e-5
use_LoRA=false
save_root=results/msmu
forget_coeff=1.0
regularization_coeff=1.0
save_checkpoint=true
save_steps=last
forget_type=formats
num_formats=5
num_forget_topics=1

for forget_loss in "${forget_losses[@]}"; do

    # model_paths setting
    model_paths=("results/llama3-8b/SGA/seed_1001/epoch5_1e-05_taskformats_5_1/1/unlearn_times_1/checkpoint-last")

    for num_epochs in "${default_epochss[@]}"; do
        for task_id in "${task_list[@]}"; do

            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                model_path=$model_path fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint forget_type=$forget_type num_formats=$num_formats num_forget_topics=$num_forget_topics"
            
            # unlearning - forget2.py
            srun --gres=gpu:2 --ntasks=1 --cpus-per-task=8 \
                torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                forget2.py --config-name=tofu.yaml task_id=$task_id save_steps=$save_steps $COMMON

            # eval_step setting
            eval_steps=(last)

            # evaluation - eval.py
            for step in "${eval_steps[@]}"; do
                srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
                    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                    eval.py --config-name=tofu.yaml task_id=$task_id eval_unlearn_step=$step $COMMON
            done
        done
    done
done
