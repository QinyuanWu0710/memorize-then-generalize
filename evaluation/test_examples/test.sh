#!/bin/bash
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres=gpu:1           # set 2 GPUs per job
#SBATCH -c 2                  # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 2-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=200GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --exclude=sws-3a100grid-01
#SBATCH -o /NS/llm-1/nobackup/qwu/lke_results/logging/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/llm-1/nobackup/qwu/lke_results/logging/%x_%j.err      # File to which STDERR will be written

model_name='llama2-13b'
relation_list=(
    7
    12
   32
   40
   50
   56
   65
   70
   73
   76
   91
   94
)

lke_list=(
    'hgp'
    'mmp'
)
index_list=(
    0
    1
    2
)

for relation in ${relation_list[@]}; do
    # /NS/venvs/work/qwu/miniconda3/envs/lke_stable/bin/python -m llm_knowledge.reliable_knowledge_estimation.main \
    #     --config_path '/NS/llm-1/work/qwu/llm_knowledge/reliable_knowledge_estimation/conf/evaluation.yaml' \
    #     --model_name $model_name \
    #     --test_relation_id $relation \
    #     --lke_type 'ic-lke' \
    #     --prompt_index 0

    for lke in ${lke_list[@]}; do
        for index in ${index_list[@]}; do
            /NS/venvs/work/qwu/miniconda3/envs/lke_stable/bin/python -m llm_knowledge.reliable_knowledge_estimation.main \
                --config_path '/NS/llm-1/work/qwu/llm_knowledge/reliable_knowledge_estimation/conf/evaluation.yaml' \
                --model_name $model_name \
                --test_relation_id $relation \
                --lke_type $lke \
                --prompt_index $index
        done
    done
done


