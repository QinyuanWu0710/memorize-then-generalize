import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
import time
import random
#use plotly to plot the distribution of the probabilities
# scatter plot, x axis is year, y axis is probability
# hover text is name
import plotly.graph_objects as go
import argparse
import os

from .eic_lke_utils import load_data, get_model_path, lke3_diff_relations, split_into_groups

from .prompt_template import HGP_TEMPLATES, MMP_TEMPLATES

#set random seed
random.seed(1)

if __name__ == "__main__":
    start_time = time.time()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="llama2-7b")
    argparser.add_argument("--tensor_parallel_size", type=int, default=2)
    argparser.add_argument("--ref_num", type=int, default=100)
    argparser.add_argument("--prefix_num", type=int, default=50)
    argparser.add_argument("--inject_num", type=int, default=10)
    argparser.add_argument("--ALL_RELATION_IDS_TO_RUN", type=str, default="50")   
    argparser.add_argument("--prompt_type", type=str, default='lke') 
    argparser.add_argument("--prompt_index", type=str, default='-1')
    
    args = argparser.parse_args()

    model_name = args.model_name
    tensor_parallel_size = args.tensor_parallel_size
    ref_num = args.ref_num
    prefix_num = args.prefix_num
    inject_num = args.inject_num
    ALL_RELATION_IDS_TO_RUN = args.ALL_RELATION_IDS_TO_RUN
    prompt_index = args.prompt_index

    ALL_RELATION_IDS_TO_RUN = [int(x) for x in ALL_RELATION_IDS_TO_RUN.split(",")]

    model_names_to_check = ['llama', 'mistral', 'vicuna', 'opt', 'openhermes', 'Nous-Hermes','inject']

    model_path = get_model_path(model_name)

    #load the model
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    if model_name.startswith('opt'):
        llm = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=2048, gpu_memory_utilization=0.9)
    else:
        llm = LLM(model=model_path, 
                  tokenizer=model_path, 
                  tensor_parallel_size=tensor_parallel_size, 
                  max_model_len=2048,
                  gpu_memory_utilization=0.5,)
    for i in range(len(ALL_RELATION_IDS_TO_RUN)):
        CHOSEN_RELATION_ID = ALL_RELATION_IDS_TO_RUN[i]
        folders = os.listdir('/NS/llm-1/work/soumidas/LKE/AlternateFacts/filtered_relations_MCQs_500')
        # remove R from the folder names
        folders = [x[1:] for x in folders]
        # intify the folder names
        folders = [int(x) for x in folders]

        df_train, df_test, chosen_relation_name = load_data(CHOSEN_RELATION_ID)

        print("Relation ID ", CHOSEN_RELATION_ID, " Relation Name ", chosen_relation_name)

        df_train['Alternate Facts'] = df_train['Alternate Facts'].apply(lambda x: eval(x))
        df_test['Alternate Facts'] = df_test['Alternate Facts'].apply(lambda x: eval(x))

        test_heads = df_test['Head'].tolist()
        if args.prompt_type != 'lke':
            if args.prompt_type == 'mmp':
                template = MMP_TEMPLATES[f"{CHOSEN_RELATION_ID}"][prompt_index]
            elif args.prompt_type == 'hgp':
                template = HGP_TEMPLATES[f"{CHOSEN_RELATION_ID}"][prompt_index]
            test_heads = [template.replace('{head}', x) for x in test_heads]
        test_facts = df_test['Fact'].tolist()
        Alternate_Facts = df_test['Alternate Facts'].tolist()
        #add test_facts to Alternate_Facts
        for j in range(len(Alternate_Facts)):
            Alternate_Facts[j].append(test_facts[j])

        train_heads = df_train['Head'].tolist()
        if args.prompt_type != 'lke':
            if args.prompt_type == 'mmp':
                template = MMP_TEMPLATES[f"{CHOSEN_RELATION_ID}"][prompt_index]
            elif args.prompt_type == 'hgp':
                template = HGP_TEMPLATES[f"{CHOSEN_RELATION_ID}"][prompt_index]
            train_heads = [template.replace('{head}', x) for x in train_heads]
        train_facts = df_train['Fact'].tolist()

        #randomly choose ref_num names from train_heads
        ref_names = random.sample(train_heads, ref_num)

        #full heads and full facts are heads and facts including both the train and test dats
        full_heads=train_heads+test_heads
        full_facts=train_facts+test_facts

        #through lke3, get the probability of each test fact
        test_head_groups = split_into_groups(test_heads, inject_num)
        result_df = pd.DataFrame()
        for m, group in enumerate(test_head_groups):
            name_begin_time = time.time()
            print(f'Group {m}')
            # Get the probability of all the alternative facts of each test name
            # print(ref_names)
            test_group_prob_dict=lke3_diff_relations(names=full_heads, 
                                        years=full_facts,
                                        ref_names=ref_names, 
                                        test_name_group=group, 
                                        alternative_years_list=Alternate_Facts[m*len(group):(m+1)*len(group)],
                                        ref_num=ref_num,
                                        prefix_num=prefix_num,
                                        inject_num=inject_num,
                                        model_path=model_path, 
                                        model_name=model_name, 
                                        llm=llm,
                                        sampling_params=sampling_params,
                                        model_names_to_check=model_names_to_check)

            # Create a list to store the data before creating DataFrame
            data_list = []

            for k, test_head in enumerate(group):
                #print(f'Adding index: all alternative facts for {test_head}: {Alternate_Facts[len(group)*m+k]}')
                #different alternative facts for each test_name
                alternative_facts =Alternate_Facts[len(group)*m+k]
                test_head_index = test_heads.index(test_head)
                truth = test_facts[test_head_index]
                #check whether truth is in alternative_facts
                #print(f'whether truth is in alternative_facts: {truth in alternative_facts}')
                for j,alternative_fact in enumerate(alternative_facts):
                    #print(alternative_fact)
                    data_list.append([test_head,truth,alternative_fact, test_group_prob_dict[test_head][j]])

            # Create a DataFrame from the list
            temp_df = pd.DataFrame(data_list, columns=['head','truth','alternative_fact', 'prob_prod'])

            # Set the multi-index
            temp_df.set_index(['head','truth','alternative_fact'], inplace=True)

            # Concatenate with the result DataFrame
            result_df = pd.concat([result_df, temp_df])
            print(result_df)
            name_end_time = time.time()
            print(f"Time for group {i}: {name_end_time - name_begin_time}")
            # Save current results to a file
            base_dir = '/NS/llm-1/work/qwu/reliable_usable_LKE/full_results/lke3/'
            #for each relation, create a folder
            if not os.path.exists(f'{base_dir}/{CHOSEN_RELATION_ID}'):
                os.makedirs(f'{base_dir}/{CHOSEN_RELATION_ID}')
            file_path = f'{base_dir}/{CHOSEN_RELATION_ID}/{model_name}_LKE3_ref-{ref_num}_prefix-{prefix_num}_inject-{inject_num}_{args.prompt_type}-{prompt_index}.parquet'
            result_df.to_parquet(file_path)


