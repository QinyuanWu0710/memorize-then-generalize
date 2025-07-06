'''
This file is used to get the subject or object probability in one sequence.
'''

import os
import pandas as pd
import yaml
import random
import json
import plotly.graph_objects as go
from vllm import SamplingParams

from .eic_lke_utils import extract_prob, get_ids
from ...util_public.inference.vllm.vllm_inference import VllmInference
from ...util_public.get_model_path import ModelPath

DATASET_PATH = '.../nobackup/qwu/dataset'
MODEL_NAMES_TO_CHECK =  ['llama', 'mistral', 'vicuna', 'opt', 'openhermes', 'Nous-Hermes']

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def plot_distribution(
        x_list: list,
        y_list: list,
        y_legend_list: list,
        title: str,
        x_label: str,
        y_label: str,
        save_path: str,
):
    fig = go.Figure()
    for i in range(len(y_list)):
        fig.add_trace(go.Scatter(x=x_list, y=y_list[i], mode='markers', name=y_legend_list[i]))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig.write_html(save_path)
    #save in png
    save_png_path = save_path.replace('.html', '.png')
    fig.write_image(save_png_path)

class GetSequenceProb:
    def __init__(
            self,
            config_path: str,
    ):
        self.config = self.load_config(config_path)
        self.config_path = config_path
        self.model_name = self.config.get('model_name')
        self.model_names_to_check = self.config.get('model_names_to_check')
        self.num_pairs = self.config.get('num_pairs')
        self.data_name = self.config.get('data_name')
        
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_data(
            self,
            data_name: str,
            num_pairs: int,
    ):
        if data_name == 'nobel':
            data_path = os.path.join(DATASET_PATH, 'nobel/0.csv')
            data_df = pd.read_csv(data_path)
            #randomly select n pairs from the data_df
            data_df = data_df.sample(n=num_pairs)
            subject_list = data_df['scientistLabel'].tolist()
            object_list = data_df['birth_year'].tolist()
            return subject_list, object_list
        elif data_name == 'trex_MC':
            pass
        #TODO: finish the trex_MC part

    def llm_generate(
            self,
            subject_list: list,
            object_list: list,
    ):
        # print(f'subject_list: {subject_list}')
        # print(f'object_list: {object_list}')
        model_path = ModelPath(self.config_path).get_model_path()
        prompt, prompt_ids, subject_ids, object_ids = get_ids(subject_list, 
                                                      object_list,
                                                      model_path,
                                                      self.model_name,
                                                      MODEL_NAMES_TO_CHECK
                                                      )
        # print(f'prompt_ids: {prompt_ids}')
        llm, sampling_params = VllmInference(self.config_path).load_model()
        #add skip_tokenizer_init = True to avoid reinitializing the tokenizer
        # print(f'prompt: {prompt}')
        # print(f'prompt_ids: {prompt_ids}')
        # print(f'subject_ids: {subject_ids}')
        # print(f'object_ids: {object_ids}')
        outputs = llm.generate(
            # prompts = prompt,
            prompt_token_ids = [prompt_ids],
            sampling_params = sampling_params,
        )

        prompt_logprobs = outputs[0].prompt_logprobs
        # print(f'prompt_logprobs: {prompt_logprobs}')
        return prompt_logprobs, subject_ids, object_ids
        

    def get_sequence_prob(
            self,
            subject_list: list,
            object_list: list,
    ):
        prompt_logprobs, subject_ids, object_ids = self.llm_generate(subject_list, object_list)
        object_probs = extract_prob(prompt_logprobs, subject_ids, object_ids)
        return object_probs
    

def main():
    config_path = '/NS/llm-1/work/qwu/llm_knowledge/util_public/inference/vllm/config/test_inference.yaml'
    get_sequence_prob = GetSequenceProb(config_path)
    subject_list, object_list = get_sequence_prob.load_data('nobel', 100)
    object_probs = get_sequence_prob.get_sequence_prob(subject_list, object_list)
    # print(object_probs)
    #plot the distribution
    x_list = list(range(len(object_probs)))
    y_list = [object_probs]
    y_legend_list = ['Object Probability']
    title = 'Object Probability Distribution'
    x_label = 'Object Index'
    y_label = 'Probability'
    save_path = '/NS/llm-1/work/qwu/llm_knowledge/reliable_knowledge_estimation/results/sequence/object_prob_distribution.html'
    plot_distribution(x_list, y_list, y_legend_list, title, x_label, y_label, save_path)

if __name__ == '__main__':
    main()