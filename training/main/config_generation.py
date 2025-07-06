'''
This file is used to generate different configurations for the training process and evaluation process.
'''

import os
import json
import yaml

def get_tokenizer_path(model_name):
    model_mapping = '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/util_public/models.json'
    with open(model_mapping, 'r') as file:
        model_map = json.load(file)
        #get the model path by model name
    model_entry = model_map[model_name]
    base_path = model_entry[1]
    model_path = os.path.join(base_path, model_entry[0])
    return model_path

def generate_evaluate_config(
        model_name_list,
        lke_type,
        tensor_parallel_size,
        save_path,
        index,
):
    #load the template config file
    with open('/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/evaluation/eval_config/evaluation.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config_list = []
    for model_name in model_name_list:
        new_config = config.copy()
        #update the model name
        new_config['model_name'] = model_name
        #split the model name
        #model_name = basemodel_relation_text_num_index_ckp
        model_name_split = model_name.split('_')
        if lke_type == 'ic-lke':
            new_config['lke_type'] = lke_type
        else:
            lke_split = lke_type.split('-')
            new_config['lke_type'] = lke_split[0]
            new_config['prompt_index'] = lke_split[1] 
        new_config['test_data_index'] = index
        new_config['tensor_parallel_size'] = tensor_parallel_size
        new_config['test_relation_id' ] = model_name_split[1]
        new_config['open_content'] = model_name_split[2]
        new_config['tokenizer_path'] = get_tokenizer_path(model_name_split[0])
        #save the new config to the save_path
        save_file = os.path.join(save_path, f'{model_name}_{lke_type}.yaml')
        with open(save_file, 'w') as file:
            yaml.dump(new_config, file)

        config_list.append(save_file)
        print(f'Config file saved at {save_file}')
    return config_list


def generate_train_config(
        train_config_template,
        train_data_index_list,
):
    #load the template config file
    try:
        with open(train_config_template, 'r') as file:
            config = yaml.safe_load(file)
        print("Config loaded from file:")
        print(config)
        if config is None:
            print("The loaded config is None, which indicates an issue with the file content or format.")
    except Exception as e:
        print(f"Error reading config: {e}")

    config_list = []
    for train_data_index in train_data_index_list:
        new_config = config.copy()
        #update the model name
        new_config['train_data_index_list'] = train_data_index
        new_config['save_label'] = f'{train_data_index[0]}-{train_data_index[-1]}'   
        #save the new config to the save_path
        #get the value of config['model_name']
        model_name = config['model_name']
        save_dir = f'/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/training/training_config/{model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        else:
            print(f'{save_dir} already exists.')
        save_file = os.path.join(save_dir, f'training_config_{train_data_index[0]}.yaml')
        #rewrite
        with open(save_file, 'w') as file:
            yaml.dump(new_config, file)
        config_list.append(save_file)
        print(f'Config file saved at {save_file}')

    return config_list, model_name


