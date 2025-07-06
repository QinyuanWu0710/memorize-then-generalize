'''
inference code using vllm
vllm documentation: https://docs.vllm.ai/en/stable/
'''
import time
import json
import yaml
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer


from ...get_model_path import ModelPath

class VllmInference:
    def __init__(self, config_path):
        """
        Initialize the VllmInference class by loading the model from the model path.
        """
        #get model loading parameters
        self.config = self.read_config(config_path)
        self.model_path = ModelPath(config_path).get_model_path()
        self.tokenizer_path = ModelPath(config_path).get_model_path()
        self.tensor_parallel_size = self.config.get('tensor_parallel_size')
        self.trust_trmote_code = self.config.get('trust_remote_code')
        self.gpu_memory_utilization = self.config.get('gpu_memory_utilization')
        self.max_model_len = self.config.get('max_model_len')
        self.prompt_logprobs = self.config.get('prompt_logprobs')
        self.batch_size = self.config.get('batch_size')
        self.max_new_tokens = self.config.get('max_new_tokens')
        self.chat_templates = self.config.get('chat_templates')
        self.skip_tokenizer_init = self.config.get('skip_tokenizer_init')
        self.temperature = self.config.get('temperature')
        self.top_k = self.config.get('top_k')
        self.top_p = self.config.get('top_p')
        self.repetition_penalty = self.config.get('repetition_penalty')
        self.system_prompt = self.config.get('system_prompt')
    

    def read_config(self, path):
        """
        Read the YAML configuration file and return the config dictionary.
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def load_tokenizer(self):
        """
        Load the tokenizer from the tokenizer path.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self):
        #before loading the model, destroy the model parallel
        # destroy_model_parallel()
        if self.prompt_logprobs is None:
            print('Prompt logprobs is not set')
            sampling_params = SamplingParams(
            max_tokens = self.max_new_tokens,
            temperature = self.temperature,
            repetition_penalty = self.repetition_penalty,
            top_k = self.top_k,
            top_p = self.top_p,
            #just for testing
            # beam_search = True,
            # early_stopping = True
        )
        else:
            sampling_params = SamplingParams(
                max_tokens = self.max_new_tokens,
                prompt_logprobs = self.prompt_logprobs,
                temperature=0
            )
        print(f'Loading model form {self.model_path}...')
        loading_start_time = time.time()
        try:
            llm = LLM(
                model = self.model_path,
                tokenizer = self.tokenizer_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self.gpu_memory_utilization, 
                trust_remote_code = self.trust_trmote_code,
                max_model_len = self.max_model_len,
            )
        except Exception as e:
            print(f'Error loading model: {e}')
            print('# NOTE: if you face OOM error for models with long context windows, like llama3 (8k) and mistral v2 (32k) you may need to try: \n\
                    # 1. you may need to decrease the max_model_len parameter to a smaller value, like 4096 \n\
                    # 2. you may need to decrease the gpu_memory_utilization parameter to a smaller value, like 0.5 \n\
                    # 3. you may need to use more GPUs to load the model (also increase the tensor_parallel_size parameter)\n\
                    # you can also check the discussion here if the above solutions do not work: https://github.com/vllm-project/vllm/issues/188\n ')
            exit()
            
        print('Model loaded.')
        print(f'Loading time: {time.time() - loading_start_time}')
        return llm, sampling_params
    
    def apply_chat_templates(self, user_prompt, system_prompt=None):
        """
        Apply chat templates for the model.
        """
        tokenizer = self.load_tokenizer()
        with open(self.chat_templates, 'r') as f:
            tokenizer.chat_template = f.read()

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": user_prompt}]
    
        full_prompt = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False,
                                                    add_generation_prompt=True, 
                                                    )
        # print(f'Full prompt: {full_prompt}')

        return full_prompt

    def inference(self, input_data, sampling_params, model):
        """
        Inference the model with the input data.
        """
        # print(f'Generating output for input: {input_data}')
        # if self.chat_templates:
        #     system_prompt = self.system_prompt #if you don't want to use system prompt, you can set it to None
        #     input_data = self.apply_chat_templates(
        #                         user_prompt=input_data, 
        #                         system_prompt=system_prompt
        #                         )

        inference_start_time = time.time()
        output = model.generate(
            input_data,
            sampling_params = sampling_params,
        )
        # print(f'Finish inference')
        # print(f'Inference time: {time.time() - inference_start_time}')
        return output
    
    def batch_inference(self, input_data, sampling_params, model):
        """
        Inference the model with the input data.
        """
        # print(f'Generating output for input: {input_data}')
        if self.chat_templates:
            system_prompt = self.system_prompt #if you don't want to use system prompt, you can set it to None
            input_data = self.apply_chat_templates(
                                user_prompt=input_data, 
                                system_prompt=system_prompt
                                )

        inference_start_time = time.time()
        output = model.generate(
            input_data,
            sampling_params = sampling_params,
        )
        print(f'Finish inference')
        print(f'Inference time: {time.time() - inference_start_time}')
        return output
    
#only for test
if __name__ == '__main__':
    config_path = '/NS/llm-1/work/qwu/llm_knowledge/util_public/inference/vllm/config/test_inference.yaml'
    inference = VllmInference(config_path)
    model, sampling_params = inference.load_model()
    def read_json_test(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        prompt = data["prompts"][0]["content"]
        return prompt
    prompt = read_json_test('/NS/llm-1/work/qwu/llm_knowledge/util_public/inference/test/chat_prompts_nl_failure-analysis.json')
    input_data = prompt
    outputs = inference.inference(input_data, sampling_params, model)
    for output in outputs:
        print(output.outputs[0].text)
