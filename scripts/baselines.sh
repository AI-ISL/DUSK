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

# NONE+GD is for Target model and Retrain model
# SGA is for TAU (make sure use TAU.sh later after completing SGA)
forget_losses=(
    NONE+GD
    GA
    GA+GD
    GA+KL
    NPO
    NPO+GD
    NPO+KL
    RMU
    TV
    SGA
)

task_list=(1)
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

default_epochss=(1 2 3 4 5)
rmu_epochss=(10 20 30 40 50)
sga_epochss=(5)

lr=1e-5
use_LoRA=false
save_root=results
forget_coeff=1.0
regularization_coeff=1.0
save_checkpoint=true
save_steps=last
forget_type=formats
num_formats=5

for forget_loss in "${forget_losses[@]}"; do

    # model_paths setting
    if [ "$forget_loss" == "NONE+GD" ]; then
        model_paths=("AI-ISL/DUSK-retrain" "AI-ISL/DUSK-target")
    else
        model_paths=("AI-ISL/DUSK-target")
    fi

    for model_path in "${model_paths[@]}"; do
        # epoch setting
        if [ "$forget_loss" == "RMU" ]; then
            epoch_list=("${rmu_epochss[@]}")
        elif [ "$forget_loss" == "NONE+GD" ]; then
            epoch_list=(1)
        elif [ "$forget_loss" == "SGA" ]; then
            epoch_list=("${sga_epochss[@]}")
        else
            epoch_list=("${default_epochss[@]}")
        fi

        for num_epochs in "${epoch_list[@]}"; do
            for task_id in "${task_list[@]}"; do

                COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr forget_loss=$forget_loss num_epochs=$num_epochs \
                    model_path=$model_path fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint forget_type=$forget_type num_formats=$num_formats"

                # unlearning - forget.py
                if [ "$forget_loss" != "NONE+GD" ]; then
                    srun --gres=gpu:2 --ntasks=1 --cpus-per-task=8 \
                        torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                        forget.py --config-name=dusk.yaml task_id=$task_id save_steps=$save_steps $COMMON
                fi

                # unlearning - forget2.py (TV)
                if [ "$forget_loss" == "TV" ]; then
                    srun --gres=gpu:2 --ntasks=1 --cpus-per-task=8 \
                        torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                        forget_tv.py --config-name=dusk.yaml task_id=$task_id save_steps=$save_steps $COMMON
                fi

                # eval_step setting
                if [ "$forget_loss" == "NONE+GD" ]; then
                    eval_steps=(0)
                else
                    eval_steps=(last)
                fi

                # evaluation - eval.py
                for step in "${eval_steps[@]}"; do
                    srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
                        torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py --config-name=dusk.yaml task_id=$task_id eval_unlearn_step=$step $COMMON
                done
            done
        done
    done
done
