'''
This script is to check the generation of each LKEs to evaluate the exact matching accuracy.
'''
import yaml
import os
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, LlamaTokenizer

from .construct_prompt import ConstructPrompt
from ...util_public.inference.vllm.vllm_inference import VllmInference
from ...util_public.get_model_path import ModelPath

import argparse

class Generation:
    '''
    Implimentationi based on vllm, check the next 50 tokens as the model response.
    '''
    def __init__(self, config_path):
        """
        Initialize the generation class by loading the configuration from a YAML file.
        """
        self.config = self.read_config(config_path)
        self.config_path = config_path
        self.lke_type = self.config.get('lke_type')
        self.model_name = self.config.get('model_name')
        self.model_path = ModelPath(config_path).get_model_path()
        self.tokenizer_path = ModelPath(config_path).get_tokenizer_path()
        self.test_dataset_name = self.config.get('test_dataset_name')
        self.test_relation_id = self.config.get('test_relation_id')
        self.num_options = self.config.get('num_options')
        self.random_seed = self.config.get('random_seed')
        self.prompt_index = self.config.get('prompt_index')
        self.test_type = self.config.get('test_type')
        self.open_content = self.config.get('open_content')
        self.example_name = self.config.get('example_name')
        self.example_num = self.config.get('example_num')
        self.example_seperator = self.config.get('example_seperator')
        self.example_reverse = self.config.get('example_reverse')
        self.save_path = self.config.get('save_path')
        self.chat_template = self.config.get('chat_template')
    
    def read_config(self, path):
        """
        Read the YAML configuration file and return the config dictionary.
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def inference(self):
        config_path = self.config_path
        prompt_constructor = ConstructPrompt(config_path)
        _, base_prompts = prompt_constructor.construct_prompt()
        #take every 100 base prompt in the list
        inputs = base_prompts

        inference_agent = VllmInference(config_path)
        model, sampling_params = inference_agent.load_model()
        outputs = model.generate(inputs, sampling_params)
        #get the output text
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)
        return responses, inputs
    
    def get_generation_acc(self, relation_id, responses):
        if self.test_dataset_name == 'trex_MC':
            df = pd.read_csv(f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/trex_MC/R{relation_id}/test_R{relation_id}.csv')
        elif self.test_dataset_name.startswith('sync_random_o'):
            df = pd.read_csv(f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/{self.test_dataset_name}/{relation_id}.csv')
        ground_truth = df['Fact'].tolist()
        #lower case for both ground_truth and response
        ground_truth = [x.lower() for x in ground_truth]
        responses = [x.lower() for x in responses]
        print(ground_truth[0], responses[0])
        #remove all the special characters like : , . etc.
        ground_truth = [x.replace(':', '').replace(',', '').replace('.', '') for x in ground_truth]
        generation_acc = []
        for i in range(len(responses)):
            if ground_truth[i] in responses[i]:
                generation_acc.append(1)
            else:
                generation_acc.append(0)
        acc = np.mean(generation_acc)
        return acc

    def save_outputs(self, relation_id, outputs, base_prompts):
        '''
        Save the outputs to a JSON file.
        '''
        acc = self.get_generation_acc(relation_id, outputs)
        save_path = self.save_path
        dict_to_save = {
            'base_prompts': base_prompts,
            'outputs': outputs,
            'generation_acc': acc
        }
        if self.test_type == 'close':
            save_name = os.path.join(save_path, 'generation', f'{self.lke_type}', f'{self.prompt_index}', f'{self.test_dataset_name}', f'{self.test_relation_id}')
        elif self.test_type == 'open':
            save_name = os.path.join(save_path, 'generation', f'{self.lke_type}', f'{self.prompt_index}', f'{self.test_dataset_name}', f'{self.test_relation_id}', f'open-{self.open_content}')
        else:
            raise ValueError(f'Unknown test type: {self.test_type}')
        
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        
        # If the model name is too long, then shorten it to only keep the last 50 characters
        if len(self.model_name) > 200:
            # split by '-'
            name_list = self.model_name.split('-')
            self.model_name = f'{name_list[0][1]}-...-{self.model_name[-200:]}'
        save_name = os.path.join(save_name, f'{self.model_name}-{self.example_name}-chat-{self.chat_template}.json')
        with open(save_name, 'w') as file:
            json.dump(dict_to_save, file)
        
        # print(f'Generation accuracy {acc}.')
        return acc, save_name

def new_config(
    config_path,
    model_name,
    relation_id,
    lke_type,
    example_name,
    example_num,
    prompt_index,
    max_new_tokens,
):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['model_name'] = model_name
    config['test_relation_id'] = relation_id
    config['max_new_tokens'] = max_new_tokens
    config['example_name'] = example_name
    config['example_num'] = example_num
    config['lke_type'] = lke_type
    config['prompt_index'] = prompt_index
    new_config_path = config_path.replace('.yaml', f'_{model_name}_{relation_id}_{lke_type}.yaml')
    # If the model name is too long, then shorten it to only keep the last 50 characters
    if len(model_name) > 100:
        # split by '-'
        name_list = model_name.split('-')
        model_name = f'{name_list[0]}-...-{model_name[-100:]}'
        new_config_path = new_config_path.replace('.yaml', f'_{model_name}.yaml')

    with open(new_config_path, 'w') as file:
        yaml.dump(config, file)
    return new_config_path

def main(
    config_path,
    model_name = 'llama2-7b',
    relation_id = 50,
    lke_type = 'ic-lke',
    example_name = 'trex_MC',
    example_num = 50,
    prompt_index = 0,
    max_new_tokens = 50,
):
    if lke_type.startswith('hgp') or lke_type.startswith('mmp'):
        #splite by '-'
        print(lke_type.split('-'))
        prompt_index = lke_type.split('-')[1]
        lke_type = lke_type.split('-')[0]
        

    config_path = new_config(
        config_path,
        model_name,
        relation_id,
        lke_type,
        example_name,
        example_num,
        prompt_index,
        max_new_tokens,
    )
    generation = Generation(config_path)
    outputs, base_prompts= generation.inference()
    generation.save_outputs(relation_id, outputs, base_prompts)
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_path', type=str, default='/NS/llm-1/work/qwu/llm_knowledge/reliable_knowledge_estimation/conf/evaluation.yaml')
    argparser.add_argument('--model_name', type=str, default='llama2-7b')
    argparser.add_argument('--test_relation_id', type=int, default=50)
    argparser.add_argument('--lke_type', type=str, default='ic-lke')
    argparser.add_argument('--example_name', type=str, default='trex_MC')
    argparser.add_argument('--example_num', type=int, default=50)
    argparser.add_argument('--prompt_index', type=int, default=0)
    argparser.add_argument('--max_new_tokens', type=int, default=50)
    args = argparser.parse_args()

    main(
        config_path = args.config_path,
        model_name = args.model_name,
        relation_id = args.test_relation_id,
        lke_type = args.lke_type,
        example_name = args.example_name,
        example_num = args.example_num,
        prompt_index = args.prompt_index,
        max_new_tokens = args.max_new_tokens,
    )