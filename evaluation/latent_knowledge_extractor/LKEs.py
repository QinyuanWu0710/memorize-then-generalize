import yaml
import os
import json
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer

from .construct_prompt import ConstructPrompt
from ...util_public.inference.vllm.vllm_inference import VllmInference
from ...util_public.get_model_path import ModelPath

class VllmLKEs:
    '''
    Implimentation based on vllm(https://blog.vllm.ai/2023/06/20/vllm.html)
    '''
    def __init__(self, config_path):
        """
        Initialize the LKEs class by loading the configuration from a YAML file.
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

    def lke_inference(self):
        config_path = self.config_path
        prompt_constructor = ConstructPrompt(config_path)
        all_input_texts, base_prompts = prompt_constructor.construct_prompt()
        print(base_prompts[-1]) #for degugging
        print(all_input_texts[-1]) #for degugging
        inference_agent = VllmInference(config_path)
        model, sampling_params = inference_agent.load_model()
        outputs = inference_agent.inference(all_input_texts, sampling_params, model)
        return outputs, base_prompts
    
    def parse_vllm_outputs(self, outputs):
        # print(outputs)
        prompt_logprobs_all = []
        for i in range(len(outputs)):
            prompt_logprobs = outputs[i].prompt_logprobs
            token_logprob_mapping = []
            for dict_log_prob in prompt_logprobs:
                if dict_log_prob:
                    # take first key value pair and add it to the mapping
                    key = list(dict_log_prob.keys())[0]
                    # token_logprob_mapping[key] = dict_log_prob[key]
                    token_logprob_mapping.append((key, dict_log_prob[key]))
            prompt_logprobs_all.append(token_logprob_mapping)
        return prompt_logprobs_all
    
    def postprocess_vllm_outputs(self, prompt_logprobs_all, base_prompts):
        # print(len(prompt_logprobs_all), len(base_prompts))
        NUM_OPTIONS = self.num_options
        # if self.tokenizer_path != None:
        #     if self.model_name.startswith('llama3'):
        #         tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        #     else:
        #         tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # else:
        #     if self.model_name.startswith('llama3'):
        #         tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        #     else:
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        ls_prompt_logprobs = [
            prompt_logprobs_all[i:i+NUM_OPTIONS]
            for i in range(0,len(prompt_logprobs_all),NUM_OPTIONS)
        ]
        encoded_base_prompts = []
        for base_prompt in base_prompts:
            encoded_base_prompts.append(tokenizer.encode(base_prompt))
        ls_prompt_logprobs_common_part_removed = []
        ls_prompt_logprobs_common_part_removed_multiplied_probs_and_normalized = []
        # print(len(ls_prompt_logprobs), len(encoded_base_prompts))
        for idx, prompt_logprobs_chunk in enumerate(ls_prompt_logprobs):
            encoded_base_prompt = encoded_base_prompts[idx]
            # copy encoded_base_prompt
            prompt_logprobs_common_part_removed = []
            for prompt_logprobs in prompt_logprobs_chunk:
                encoded_base_prompt_copy = encoded_base_prompt.copy()
                # remove the first token from the encoded_base_prompt_copy
                encoded_base_prompt_copy = encoded_base_prompt_copy[1:]
                # print(encoded_base_prompt_copy)
                # print(prompt_logprobs)
                new_out = prompt_logprobs[len(encoded_base_prompt_copy):]
                # apply np.exp to the logprobs
                # new_out = [(k, v) for k,v in new_out]
                # Instead of the above, use logsumexp trick later
                prompt_logprobs_common_part_removed.append(new_out)
            ls_prompt_logprobs_common_part_removed.append(prompt_logprobs_common_part_removed)
        ls_prompt_logprobs_common_part_removed_multiplied_probs = []
        for prompt_logprobs_chunk in ls_prompt_logprobs_common_part_removed:
            prompt_logprobs_common_part_removed_multiplied_probs = []
            # print(prompt_logprobs_chunk)
            for prompt_logprobs in prompt_logprobs_chunk:
                # print(prompt_logprobs)
                values = [x[1] for x in prompt_logprobs]
                # The output is updated in latest vllm version
                # if values[0] is float, then it is logprob
                # if isinstance(values[0], float):
                #     values = [x for x in values]
                # else:
                values = [x.logprob for x in values]
                prompt_logprobs_common_part_removed_multiplied_probs.append(np.exp(sum(values)))
            ls_prompt_logprobs_common_part_removed_multiplied_probs.append(prompt_logprobs_common_part_removed_multiplied_probs)
            # normalize the list
            # print(prompt_logprobs_common_part_removed_multiplied_probs)
            sum_prompt_logprobs_common_part_removed_multiplied_probs = sum(prompt_logprobs_common_part_removed_multiplied_probs)
            prompt_logprobs_common_part_removed_multiplied_probs_and_normalized = [i/sum_prompt_logprobs_common_part_removed_multiplied_probs for i in prompt_logprobs_common_part_removed_multiplied_probs]
            ls_prompt_logprobs_common_part_removed_multiplied_probs_and_normalized.append(prompt_logprobs_common_part_removed_multiplied_probs_and_normalized)
            # print(prompt_logprobs_common_part_removed_multiplied_probs)
        return ls_prompt_logprobs_common_part_removed_multiplied_probs, ls_prompt_logprobs_common_part_removed_multiplied_probs_and_normalized

    def evaluate(self, post_processed_outputs):
        correct = 0
        prob_mass_correct_answer = 0
        for idx, prompt_logprobs_chunk in enumerate(post_processed_outputs):
            # find highest prob idx
            highest_prob_idx = prompt_logprobs_chunk.index(max(prompt_logprobs_chunk))
            # print(highest_prob_idx, len(prompt_logprobs_chunk))
            if highest_prob_idx == len(prompt_logprobs_chunk) - 1:
                correct = correct + 1
            prob_mass_correct_answer = prob_mass_correct_answer + prompt_logprobs_chunk[-1]
        return correct/len(post_processed_outputs), prob_mass_correct_answer/len(post_processed_outputs)

    def save_out(self, base_prompts, post_processed_outputs, post_processed_outputs_normalized, accuracy, prob_mass):
        #save folder is defined by lke_type-prompt_index-test_model-test_dataset-test_relation_id
        save_folder = f'{self.save_path}/{self.lke_type}/{self.prompt_index}/{self.model_name}/{self.test_dataset_name}/{self.test_relation_id}'
        #if the model name is too long, then shorten it to only keep the last 50 characters
        if len(self.model_name) > 200:
            # split by '-'
            name_list = self.model_name.split('-')
            self.model_name = f'{name_list[0][1]}-...-{self.model_name[-200:]}'
            save_folder = f'{self.save_path}/{self.lke_type}/{self.prompt_index}/{self.model_name}/{self.test_dataset_name}/{self.test_relation_id}'
        #if the save folder does not exist, then create it
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
        #save all the outputs into random_seed-{random_seed}.json
        if self.test_type == 'close':
            save_path = f'{save_folder}/random_seed-{self.random_seed}-{self.example_seperator}-{self.example_name}-chat-{self.chat_template}.json'
        elif self.test_type == 'open':
            save_path = f'{save_folder}/random_seed-{self.random_seed}-{self.example_seperator}-{self.example_name}-chat-{self.chat_template}-open-{self.open_content}.json'
        else:
            raise ValueError(f'Unknown test type: {self.test_type}')

        dict_to_save = {
            'base_prompts': base_prompts,
            'post_processed_outputs': post_processed_outputs,
            'post_processed_outputs_normalized': post_processed_outputs_normalized,
            'accuracy': accuracy,
            'prob_mass': prob_mass
        }
        with open(save_path, 'w') as f:
            json.dump(dict_to_save, f)
   
    def get_accuracy(self):
        #get outputs based on the config
        outputs, base_prompts = self.lke_inference()
        prompt_logprobs_all = self.parse_vllm_outputs(outputs)
        prompt_logprobs_common_part_removed_multiplied_probs, prompt_logprobs_common_part_removed_multiplied_probs_and_normalized = self.postprocess_vllm_outputs(prompt_logprobs_all, base_prompts)
        accuracy, prob_mass_correct_answer = self.evaluate(prompt_logprobs_common_part_removed_multiplied_probs_and_normalized)
        self.save_out(
            base_prompts, 
            prompt_logprobs_common_part_removed_multiplied_probs,
            prompt_logprobs_common_part_removed_multiplied_probs_and_normalized, 
            accuracy, 
            prob_mass_correct_answer
            )
        return accuracy, prob_mass_correct_answer
    
