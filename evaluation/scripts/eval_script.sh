#!/bin/bash 
#SBATCH --partition a40,a100,h100           # Use GPU partition
#SBATCH --nodes=1                   # Request 2 nodes
#SBATCH -N 1
#SBATCH --ntasks=1                  # Total number of tasks (one per node)
#SBATCH --gres=gpu:1                # Request 6 GPUs per node
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 01-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=200GB               # Memory pool for all cores (see also --mem-per-cpu)

#SBATCH -o /NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/logging/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/logging/%x_%j.err      # File to which STDERR will be written


export GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export GPUS_PER_NODE=$(scontrol show job $SLURM_JOB_ID | grep -oP "(?<=GRES=gpu:)\d+")

test_dataset_name="sync_random_o"          
test_index_begin=0
test_index_end=100

test_type='close'
max_new_tokens=100


lke_types=(
     'test-1'
)
relation_id="50" #test relation id


for lke_type in "${lke_types[@]}"; do
    model_names=(
                "qwen2.5-1.5b"
    )

    for model_name in "${model_names[@]}"; do
    # print the model name
        echo "Test model_name: $model_name"
        #after training run the following command to evaluate the model
        python -m code_open.evaluation.eval \
            --eval_config_path '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/conf/_eval_config.yaml' \
            --model_name "$model_name" \
            --test_dataset_name "$test_dataset_name" \
            --example_name "sync_random_o" \
            --relation_id "$relation_id" \
            --lke_type "$lke_type"  \
            --num_options 100  \
            --test_index_begin "$test_index_begin" \
            --test_index_end "$test_index_end" \
            --open_content 'hgp-0' \
            --test_type "${test_type}"

        python -m code_open.evaluation.eval_gen \
            --eval_config_path '/NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/evaluation/conf/_eval_gen.yaml' \
            --model_name "$model_name" \
            --test_dataset_name "$test_dataset_name" \
            --example_name "sync_random_o" \
            --relation_id "$relation_id" \
            --lke_type "$lke_type"  \
            --num_options 100  \
            --max_new_tokens "$max_new_tokens" \
            --test_index_begin "$test_index_begin" \
            --test_index_end "$test_index_end" \
            --open_content 'hgp-0' \
            --test_type  "${test_type}"
    done
done