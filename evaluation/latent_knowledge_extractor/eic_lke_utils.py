import os
import random
import time
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

#use plotly to plot the distribution of the probabilities
# scatter plot, x axis is year, y axis is probability
# hover text is name
import plotly.graph_objects as go
import argparse
import os

def get_model_path(model_name):
    BASE_MODEL_DIR_VEDANT = "/NS/llm-1/nobackup/vnanda/llm_base_models"
    FT_MODEL_DIR_VEDANT = "/NS/llm-1/nobackup/vnanda/llm_finetuning"
    FT_MODEL_DIR_QWU = "/NS/llm-1/nobackup/qwu/llm_finetuning"
    BASE_MODEL_DIR_QWU_12B = "/NS/llm-1/nobackup/qwu/llm_base_model/pythia-12b"
    BASE_MODEL_DIR_QWU = "/NS/llm-1/nobackup/qwu/llm_base_model"
    BASE_MODEL_DIR_CXU = '/NS/llm-1/nobackup/cxu/llm_base_models'
    BASE_MODEL_DIR_AFLAH = '/NS/llm-1/nobackup/afkhan/Model_Saves'
    BASE_MODEL_DIR_SOUMI = '/NS/llm-1/nobackup/soumi'
    INJECT_MODEL_DIR = "/NS/llm-1/nobackup/vnanda/llm_finetuning/continual_training_wikipedia/llama2-7b-birthyear-pdbs5-schedule-cosine-warmup-0"
    INJECT_MODEL_DIR_QWU = "/NS/llm-1/work/qwu/reliable_usable_LKE/sanity_check/models/saves-pdbs5-schedule-cosine-warmup-0"
    INJECT_LLAMA_76 = "/NS/llm-1/work/qwu/reliable_usable_LKE/sanity_check/saves/76-pdbs5-schedule-cosine-warmup-0"
    INJECT_LLAMA_76_HGP = "/NS/llm-1/work/qwu/reliable_usable_LKE/sanity_check/saves/hgp-76-pdbs5-schedule-linear-warmup-80"
    MODELS = {
        "gemma-2b": ("gemma-2b", BASE_MODEL_DIR_QWU),
        "gemma-2b-it": ("gemma-2b-it", BASE_MODEL_DIR_QWU),
        "gemma-7b": ("gemma-7b", BASE_MODEL_DIR_QWU),
        "gemma-7b-it": ("gemma-7b-it", BASE_MODEL_DIR_QWU),
        "dbrx-base": ("dbrx-base", BASE_MODEL_DIR_VEDANT),
        "dbrx-instruct":("databricks--dbrx-instruct", BASE_MODEL_DIR_AFLAH),
        "pyt-70m": ("pythia-70m", BASE_MODEL_DIR_VEDANT),
        "pyt-160m": ("pythia-160m", BASE_MODEL_DIR_VEDANT),
        "pyt-410m": ("pythia-410m", BASE_MODEL_DIR_VEDANT),
        "pyt-1b": ("pythia-1b", BASE_MODEL_DIR_VEDANT),
        "pyt-1.4b": ("pythia-1.4b", BASE_MODEL_DIR_VEDANT),
        "pyt-2.8b": ("pythia-2.8b", BASE_MODEL_DIR_VEDANT),
        # "pyt-2.8b": ("EleutherAI/pythia-2.8b", None),
        "pyt-6.9b": ("pythia-6.9b", BASE_MODEL_DIR_VEDANT),
        "pyt-12b": ("pythia-12b", BASE_MODEL_DIR_VEDANT),
        "pyt-12b-ckp0": ("step-0", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp1000": ("step-1000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp3000": ("step-3000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp5000": ("step-5000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp7000": ("step-7000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp10000": ("step-10000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp20000": ("step-20000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp30000": ("step-30000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp40000": ("step-40000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp50000": ("step-50000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp60000": ("step-60000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp70000": ("step-70000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp80000": ("step-80000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp90000": ("step-90000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp100000": ("step-100000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp110000": ("step-110000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp120000": ("step-120000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp130000": ("step-130000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp140000": ("step-140000", BASE_MODEL_DIR_QWU_12B),
        "pyt-12b-ckp143000": ("step-143000", BASE_MODEL_DIR_QWU_12B),
        "llama2-7b": ("Llama-2-7b-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-13b": ("Llama-2-13b-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-70b": ("Llama-2-70b-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-7b-chat": ("Llama-2-7b-chat-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-13b-chat": ("Llama-2-13b-chat-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-70b-chat": ("Llama-2-70b-chat-hf", BASE_MODEL_DIR_VEDANT),
        "llama2-ft-7b": ("llama2-7b-dolly-finetune-pdbs4", FT_MODEL_DIR_VEDANT),
        "gpt2": ("gpt2", BASE_MODEL_DIR_SOUMI),
        "gpt2-xl": ("gpt2-xl", BASE_MODEL_DIR_SOUMI),
        "pyt-ft-70m": ("pythia-70m-finetuning-pdbs16", FT_MODEL_DIR_VEDANT),
        "pyt-ft-160m": ("pythia-160m-finetuning-pdbs16", FT_MODEL_DIR_VEDANT),
        "pyt-ft-410m": ("pythia-410m-finetuning-pdbs16", FT_MODEL_DIR_VEDANT),
        "pyt-ft-1b": ("pythia-1b-finetuning-pdbs4-gacc4", FT_MODEL_DIR_VEDANT),
        "pyt-ft-1.4b": ("pythia-1.4b-finetuning-pdbs8", FT_MODEL_DIR_VEDANT),
        "pyt-ft-2.8b": ("pythia-2.8b-finetuning", FT_MODEL_DIR_QWU),
        # "pyt-ft-6.9b":("pythia-6.9b-finetuning-pdbs4", FT_MODEL_DIR_VEDANT),
        "pyt-ft-12b": ("pythia-12b-finetuning-pdbs16", FT_MODEL_DIR_VEDANT),
        "gpt-neox-20b": ("gpt-neox-20b", BASE_MODEL_DIR_QWU),
        "baichuan2-7b": ("baichuan2-7b", BASE_MODEL_DIR_QWU),
        "mistral-7b": ("Mistral-7B-v0.1", BASE_MODEL_DIR_VEDANT),
        "falcon-7b":("falcon-7b", BASE_MODEL_DIR_QWU),
        "llama-7b": ("llama_7b", BASE_MODEL_DIR_QWU),
        "llama-13b": ("llama_13b", BASE_MODEL_DIR_QWU),
        "llama-33b": ("llama_30b", BASE_MODEL_DIR_QWU),
        "llama-65b": ("llama_65b", BASE_MODEL_DIR_CXU),
        "vicuna-13b": ("lmsys--vicuna-13b-v1.5", BASE_MODEL_DIR_AFLAH),
        "vicuna-13b-llama-original": ("lmsys--vicuna-13b-v1.3", BASE_MODEL_DIR_AFLAH),
        "vicuna-7b": ("lmsys--vicuna-7b-v1.5", BASE_MODEL_DIR_AFLAH),
        "vicuna-7b-llama-original": ("lmsys--vicuna-7b-v1.3", BASE_MODEL_DIR_AFLAH),
        "opt-125m": ("opt-model-125m", BASE_MODEL_DIR_SOUMI),
        "opt-350m": ("opt-model-350m", BASE_MODEL_DIR_SOUMI),
        "opt-1.3b": ("opt-model-1.3B", BASE_MODEL_DIR_SOUMI),
        "opt-2.7b": ("opt-model-2.7B", BASE_MODEL_DIR_SOUMI),
        "opt-6.7b": ("opt-model-6.7B", BASE_MODEL_DIR_SOUMI),
        "opt-13b": ("opt-model-13B", BASE_MODEL_DIR_SOUMI),
        "opt-30b": ("opt-model-30B", BASE_MODEL_DIR_SOUMI),
        "opt-66b": ("facebook--opt-66b", BASE_MODEL_DIR_AFLAH),
        "bloom-560m": ("bigscience--bloom-560m", BASE_MODEL_DIR_AFLAH),
        "bloom-1.1b": ("bigscience--bloom-1b1", BASE_MODEL_DIR_AFLAH),
        "bloom-1.7b": ("bigscience--bloom-1b7", BASE_MODEL_DIR_AFLAH),
        "bloom-3b": ("bigscience--bloom-3b", BASE_MODEL_DIR_AFLAH),
        "bloom-7.1b": ("bigscience--bloom-7b1", BASE_MODEL_DIR_AFLAH),
        "bloom-176b": ("bigscience--bloom", BASE_MODEL_DIR_AFLAH),
        "gpt-j-6b": ("gpt-j-6b", BASE_MODEL_DIR_VEDANT),
        "gpt-neox-20b": ("gpt-neox-20b", BASE_MODEL_DIR_QWU),
        "falcon-40b": ("tiiuae--falcon-40b", BASE_MODEL_DIR_AFLAH),
        "falcon-180b": ("tiiuae--falcon-180B", BASE_MODEL_DIR_AFLAH),
        "falcon-instruct-7b": ("tiiuae--falcon-7b-instruct", BASE_MODEL_DIR_AFLAH),
        "falcon-instruct-40b": ("tiiuae--falcon-40b-instruct", BASE_MODEL_DIR_AFLAH),
        "falcon-instruct-180b": ("falcon-180b-chat", BASE_MODEL_DIR_QWU),
        "mistral-instruct-7b": ("Mistral-7B-Instruct-v0.1", BASE_MODEL_DIR_VEDANT),
        "openhermes-2.5-mistral-7b": ("teknium--OpenHermes-2.5-Mistral-7B", BASE_MODEL_DIR_AFLAH),
        "phi-2": ("microsoft--phi-2", BASE_MODEL_DIR_AFLAH),
        "mistral-mixtral-8x7B-v0.1": ("mistralai--Mixtral-8x7B-v0.1", BASE_MODEL_DIR_AFLAH),
        "openchat-3.5-0106": ("openchat--openchat-3.5-0106", BASE_MODEL_DIR_AFLAH),
        "Nous-Hermes-2-Mixtral-8x7B-SFT": ("NousResearch--Nous-Hermes-2-Mixtral-8x7B-SFT", BASE_MODEL_DIR_AFLAH),
        "Nous-Hermes-2-Mixtral-8x7B-DPO": ("NousResearch--Nous-Hermes-2-Mixtral-8x7B-DPO", BASE_MODEL_DIR_AFLAH),
        "Amber": ("LLM360--Amber", BASE_MODEL_DIR_AFLAH),
        "mpt-7b": ("mosaicml--mpt-7b", BASE_MODEL_DIR_AFLAH),

        "inject-llama2-7b-76_ckp-40": ('checkpoint-40', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-80": ('checkpoint-80', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-120": ('checkpoint-120', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-160": ('checkpoint-160', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-200": ('checkpoint-200', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-240": ('checkpoint-240', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-280": ('checkpoint-280', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-320": ('checkpoint-320', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-360": ('checkpoint-360', INJECT_LLAMA_76),
        "inject-llama2-7b-76_ckp-400": ('checkpoint-400', INJECT_LLAMA_76),

        "inject-llama2-7b-hgp-76_ckp-40": ('checkpoint-40', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-80": ('checkpoint-80', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-120": ('checkpoint-120', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-160": ('checkpoint-160', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-200": ('checkpoint-200', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-240": ('checkpoint-240', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-280": ('checkpoint-280', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-320": ('checkpoint-320', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-360": ('checkpoint-360', INJECT_LLAMA_76_HGP),
        "inject-llama2-7b-hgp-76_ckp-400": ('checkpoint-400', INJECT_LLAMA_76_HGP),

}
    model_entry = MODELS[model_name]
    base_path = model_entry[1]
    model_path = os.path.join(base_path, model_entry[0])
    return model_path

def extract_prob(prompt_logprobs, name_ids, year_ids):
    #get the probability of each test name
    year_probs = []
    for k in range(len(year_ids)):
        year_prob = 0
        for i in range(len(year_ids[k])):
            if k==0:
                begin_index = len(name_ids[k])+1
                # print(year_ids[k][i])
                # print(prompt_logprobs[begin_index+i])
            else:
                begin_index = 0
                for j in range(k+1):
                    begin_index += len(name_ids[j])+1
                for m in range(k):
                    begin_index += len(year_ids[m])+1
                # print(year_ids[k][i])
                # print(prompt_logprobs[begin_index+i])
            #if year_ids[k][i] is not the key of prompt_logprobs[begin_index+i], then the probability is 0
                        #if begin_index+i is out of range, then the probability is 0
            if begin_index+i >= len(prompt_logprobs):
                year_prob += -100
                # print('out of range')
                # print(year_ids[k][i])
                # print(prompt_logprobs[begin_index+i])
            elif year_ids[k][i] not in prompt_logprobs[begin_index+i].keys():
                year_prob += -100
                # print('not in keys')
                # print(year_ids[k][i])
                # print(prompt_logprobs[begin_index+i])
            else:
                prob = prompt_logprobs[begin_index+i][year_ids[k][i]]
                if isinstance(prob, float):
                    prob = prob
                else:
                    prob = prob.logprob
                year_prob += prob
                # print(year_ids[k][i])
                # print(prompt_logprobs[begin_index+i])
        year_probs.append(year_prob)

    #calculate the exponential of each year's probability
    year_probs_exp = []
    for year_prob in year_probs:
        year_probs_exp.append(np.exp(year_prob))
    return year_probs_exp

def load_data(relation_id):
    base_path = '/NS/llm-1/work/soumidas/LKE/AlternateFacts/filtered_relations_MCQs_500_details'
    meta_data_path = '/NS/llm-1/work/soumidas/LKE/AlternateFacts/'
    no_mask_file = meta_data_path + 'all_relationID_names_nomask.json'
    df_relation_data = pd.read_json(no_mask_file)
    df_relation_data = df_relation_data.T
    relations = df_relation_data[1].tolist()
    chosen_relation_name = relations[relation_id-1]
    chosen_relation = 'R' + str(relation_id)
    relation_data_folder_path = base_path + '/' + chosen_relation + '/'
    train_file_name = relation_data_folder_path + f'train_{chosen_relation}.csv'
    test_file_name = relation_data_folder_path + f'test_{chosen_relation}.csv'
    df_train = pd.read_csv(train_file_name)
    df_test = pd.read_csv(test_file_name)
    return df_train, df_test, chosen_relation_name

#set random seed
random.seed(1)

def delete_BOS(id, model_name, model_names_to_check):
    #for those models which added BOS token, delete the BOS token
    #if model name start with 'llama', 'mistral', then delete the first token
    for name in model_names_to_check:
        if model_name.startswith(name):
            del id[0]
    return id

def get_ids(names, years, model_path, model_name, model_names_to_check):
    #tokenize all the names and years, store the token ids in lists
    #tokenizer = AutoTokenizer.from_pretraicned(model_path, skip_special_tokens=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    name_ids = []
    year_ids = []
    # print(f'names: {names}')
    # print(f'years: {years}')
    for name in names:
        name_id = tokenizer.encode(name)
        name_id = delete_BOS(name_id, model_name, model_names_to_check)
        name_ids.append(name_id)
    for year in years:
        #for numbers, specifically delete the first two tokens
        year_id = tokenizer.encode(str(year))
        if model_name.startswith('opt'):
            year_id = delete_BOS(year_id, model_name, model_names_to_check)
        else:
            # year_id = delete_BOS(year_id, model_name, model_names_to_check)
            year_id = delete_BOS(year_id, model_name, model_names_to_check)
        year_ids.append(year_id)
    #     print(f'year_id: {year_id}')
    #     print(f'year: {tokenizer.decode(year_id)}')
    # print(f'name_ids: {name_ids}')
    # print(f'year_ids: {year_ids}')
    
    for name in model_names_to_check:
        if model_name.startswith(name):
            space_token_id = tokenizer.encode(' ')[1]
            comma_token_id = tokenizer.encode(',')[1]
            break
        else:
            space_token_id = tokenizer.encode(' ')[0]
            comma_token_id = tokenizer.encode(',')[0]
    
    #test prompt should be 'name1 birthyear1, name2 birthyear2, ...', force the tokenization to be the same as name_ids and year_ids
    test_prompt_ids = []
    for i in range(len(names)):
        test_prompt_ids += name_ids[i]
        test_prompt_ids.append(space_token_id)
        test_prompt_ids += year_ids[i]
        test_prompt_ids.append(comma_token_id)

    prompt = tokenizer.decode(test_prompt_ids)
    # print(f'prompt: {prompt}')
    return prompt, test_prompt_ids, name_ids, year_ids

def llm_generate(batch_names, batch_years, model_path, model_name, llm, sampling_params, model_names_to_check):
    name_ids_list = []
    year_ids_list = []
    test_prompt_ids_list = []
    for test_names, test_years in zip(batch_names, batch_years):
        #print(f'test_names: {test_names}')
        test_prompt_ids, name_ids, year_ids = get_ids(test_names, test_years, model_path, model_name, model_names_to_check)
        name_ids_list.append(name_ids)
        year_ids_list.append(year_ids)
        test_prompt_ids_list.append(test_prompt_ids)
    # print(f"test_prompt_ids_list: {len(test_prompt_ids_list)}")
    # print(f'example test_prompt_ids: {test_prompt_ids_list[0]}')
    # exit()
    outputs = llm.generate(prompt_token_ids=test_prompt_ids_list, sampling_params=sampling_params)
    #print(f"outputs: {outputs}")
    #get prompt logprobs
    prompt_logprobs_list = [outputs[i].prompt_logprobs for i in range(len(outputs))]
    # print(f"prompt_logprobs_list[1]: {prompt_logprobs_list[1]}")
    return prompt_logprobs_list, name_ids_list, year_ids_list

def insert_alternative_facts(ref_names: list, 
                             ref_years: list, 
                             test_name_group: list,
                             alternative_years: list,
                             ref_num: int,
                             prefix_num: int,
                             inject_num: int):
    # after 20 examples, insert alternative facts every 5 examples
    # randomly choose one year from 1800 to 2000 as the alternative year
    # inject (test_name, alternative_year) into the list every 5 examples
    # store the inject index
    inject_index = []
    inject_names = []
    inject_years = []
    j=0
    seperate_num = (ref_num-prefix_num)//inject_num
    for i in range(len(ref_names)):
        if i >= prefix_num and i%seperate_num == 0:
            inject_index.append(i)
            #to make sure all the alternative facts have different head entities and tail entities in one sequence
            inject_names.append(test_name_group[j])
            inject_years.append(alternative_years[j])
            j+=1
        else:
            inject_names.append(ref_names[i])
            inject_years.append(ref_years[i])
    
    return inject_names, inject_years, inject_index

def move_list(list):
    #move all the elements in the list to the right by one position, the last element becomes the first element
    temp = list[-1]
    for i in range(len(list)-1, 0, -1):
        list[i] = list[i-1]
    list[0] = temp
    return list 

def get_year_info(names, years, ref_names):
    #get corresponding birth years
    ref_years = []
    for name in ref_names:
        # Find the index of each name in the 'names' list
        index = names.index(name)
        # Append the corresponding year with a comma to test_years
        ref_years.append(str(years[index]))
    return ref_years

def split_into_groups(lst, n):
    # Splitting the list into groups of 10
    groups = [lst[i:i + n] for i in range(0, len(lst), n)]
    
    # Removing the last group if it has less than n items
    if len(groups[-1]) < n:
        groups.pop()

    return groups

def lke3_diff_relations(names: list, 
        years: list,
        ref_names: list, 
        test_name_group: list, 
        #alternative_years is a list of lists
        alternative_years_list: list,
        ref_num: int,
        prefix_num: int,
        inject_num: int,
        model_path: str, 
        model_name: str, 
        llm: LLM, 
        sampling_params: SamplingParams,
        model_names_to_check: list):
    # print('start lke3')
    #for a group of test names, get the probability of all the alternative facts of each test name
    ref_years = get_year_info(names, years, ref_names)
    #the key is every test name in this group, the value is the probability list of all the alternative facts of this test name
    #get the probability of each test name in this group for the alternative facts
    batch_names =[]
    batch_years = []
    batch_index = []
    batch_test_name = []
    test_name_prob_dict = {test_name: [0]*len(alternative_years_list[0]) for test_name in test_name_group}
    
    original_test_name_group = test_name_group.copy()
    alternative_years_inject_group_list = []
    for x in range(len(alternative_years_list)):
        alternative_years_inject_group_list.append(split_into_groups(alternative_years_list[x], inject_num))

    # print(f'alternative_years_inject_group_list: {alternative_years_inject_group_list}')
    # #show the shape of alternative_years_inject_group_list
    # print(f'alternative_years_inject_group_list shape: {len(alternative_years_inject_group_list)}')
    # print(f'alternative_years_inject_group_list shape: {len(alternative_years_inject_group_list[0])}')
    # print(f'alternative_years_inject_group_list shape: {len(alternative_years_inject_group_list[0][0])}')
    # print(f'alternative_years_inject_group_list shape: {alternative_years_inject_group_list[0][0][0]}')

    #print(f'alternative_years_inject_group_list: {alternative_years_inject_group_list}')
    for j in range(len(test_name_group)):
        #every test name's alternative facts are different
        #print(f'alternative years for {test_name_group[j]}: {alternative_years_inject_group_list[j]}')
        for i in range(len(alternative_years_inject_group_list[0])):      
            #take first group of each test name as a list of inject alternatives
            inject_alternatives = []
            for y in range(len(test_name_group)):   
                #MOST tricky part, ensure injec the correct year for each test name
                #since each test name got different alternative fatcs
                #choose jth head's ith group
                inject_alternatives.append(alternative_years_inject_group_list[y][i][y])
            #print(f'inject_alternatives: {inject_alternatives}')
            inject_names, inject_years, inject_index = insert_alternative_facts(ref_names,ref_years, test_name_group, inject_alternatives, ref_num,prefix_num, inject_num)
            batch_names.append(inject_names)
            batch_years.append(inject_years)
            batch_index.append(inject_index)
            batch_test_name.append(test_name_group)
        test_name_group = move_list(test_name_group)
        alternative_years_inject_group_list = move_list(alternative_years_inject_group_list)

    #do inference for all the possible alternative facts
    prompt_logprobs_list, name_ids_list, year_ids_list = llm_generate(batch_names, batch_years, model_path, model_name, llm, sampling_params, model_names_to_check)
    # print(f'test_name_group:{batch_names[0]}')
    # divide all the list back to groups
    #get the probability of each test name
     #Initialization: set a blank list for each test name

    for j,test_name in enumerate(original_test_name_group): #9
       # print(f'original test name group:{original_test_name_group}')
        for i in range(len(alternative_years_inject_group_list[0])):#19
            index_of_test_name_in_batch = j*len(alternative_years_inject_group_list[0])+i
            year_probs_exp = extract_prob(prompt_logprobs_list[index_of_test_name_in_batch], name_ids_list[index_of_test_name_in_batch], year_ids_list[index_of_test_name_in_batch])
            # print(f'year_probs_exp:{year_probs_exp}')
            # exit()
            index = batch_index[index_of_test_name_in_batch]
            for k in range(len(index)):
                test_name_prob_dict[original_test_name_group[k]][k+len(original_test_name_group)*i] = year_probs_exp[index[k]]
        original_test_name_group = move_list(original_test_name_group)

    return test_name_prob_dict

def plot(distribution, inject_index,i):
    #x axis is index from 0 to 199, y axis is probability
    #inject_index is the index of the alternative facts
    #plot the distribution of the probabilities
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(distribution))), y=distribution, mode='markers'))
    #for scatter of inject_index, set the color to red
    fig.add_trace(go.Scatter(x=inject_index, y=[distribution[i] for i in inject_index], mode='markers', marker_color='red'))
    #save_fig to intermedia file /NS/llm-1/work/qwu/knowledge_probing/src/experiments/LKE-3/result/test/{i}_test.png
    fig.write_image(f'/NS/llm-1/work/qwu/knowledge_probing/src/experiments/LKE-3/result/test/{i}_test.png')
    return 0