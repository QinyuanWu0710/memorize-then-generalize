#!/bin/bash
#SBATCH --partition=a100,h100,h200,a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2           # set 2 GPUs per job
#SBATCH -c 16                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 01-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=400GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-3a100grid-01,sws-8a100-03
#SBATCH -o /NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/logging/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/logging/%x_%j.err      # File to which STDERR will be writte

export GPUS_PER_NODE=2
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3373

relation_id=50

#'llama2-7b_50_trex_MC_hgp-7_s-o_1_trex_MC-50_2e-6_0-49_checkpoint-13'
# Using accelerate to launch the module
accelerate launch --config_file="/NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/util_public/training/config/sft_ds_z2.yaml" \
    --num_processes $GPUS_PER_NODE \
    --main_process_port $MASTER_PORT \
    -m code_open.training.main.pipeline_train_custome \
    --train_template_config_path "/NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/training/conf/training_config_custome.yaml" \
    --base_model_name "qwen2.5-1.5b"\
    --relation_id "${relation_id}" \
    --data_name "sync_random_o" \
    --train_text_type 'train-mix' \
    --loss_computation 'normal'\
    --inject_facts_num 100 \
    --old_facts_num 0 \
    --index_begin 0 \
    --index_end 100 \
    --example_num 50 \
    --example_name "trex_MC" \
    --epoch 1 \
    --learning_rate 5e-06 \
    --lr_scheduler_type 'cosine' 
    # --chat_template '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/util_public/inference/chat_templates/llama2.jinja'
