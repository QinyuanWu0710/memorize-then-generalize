import gc
import torch
import yaml
import os
import argparse

# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
#for vllm > 4.0.0
from vllm.distributed.parallel_state import destroy_model_parallel
from ..evaluation.latent_knowledge_extractor.generation import Generation
#use the config to do evaluation 
def eval(
        config_path,
        model_name,
        test_dataset_name,
        num_options,
        relation_id,
        lke_type,
        example_name,
        test_index_begin=None,       
        test_index_end=None,
        max_new_tokens=1, 
        chat_template=False,
        # tokenizer_path=None
        test_type='open',
        open_content=None
):
    train_config = yaml.safe_load(open(config_path, 'r'))
    train_config['model_name'] = model_name
    train_config['test_dataset_name'] = test_dataset_name
    train_config['lke_type'] = lke_type
    train_config['test_relation_id'] = relation_id
    train_config['example_name'] = example_name
    train_config['num_options'] = num_options
    train_config['max_new_tokens'] = max_new_tokens
    train_config['chat_template'] = chat_template
    train_config['open_content'] = open_content
    train_config['test_type'] = test_type
    # train_config['tokenizer_path'] = tokenizer_path
    config_file_name = f'{model_name}_{lke_type}_eval_config.yaml'
    # If it's too long, then shorten the model name to only keep the last 50 characters
    if len(model_name) > 200:
        # split by '-'
        name_list = model_name.split('-')
        model_name = f'{name_list[0]}-...-{model_name[-200:]}'
    config_file_name = f'{model_name}_{lke_type}_eval_config.yaml'
    new_config_path = os.path.join(os.path.dirname(config_path), config_file_name)

    test_config = train_config
    if test_index_begin is not None:
        # test_config = yaml.safe_load(open(new_config_path, 'r'))
        test_config['test_data_type'] = 'test'
        test_config['test_index_begin'] = test_index_begin
        test_config['test_index_end'] = test_index_end
        if lke_type.startswith('hgp') or lke_type.startswith('mmp') or lke_type.startswith('test') or lke_type.startswith('train') or lke_type.startswith('NegTrain') or lke_type.startswith('NegTest'):
            lke_type, prompt_index = lke_type.split('-')
            train_config['lke_type'] = lke_type
            train_config['prompt_index'] = int(prompt_index)

        #write the config to new temp config
        with open(new_config_path, 'w') as file:
            yaml.dump(test_config, file)

        gen_lkes = Generation(new_config_path)
        # print('start')
        outputs, base_prompts = gen_lkes.inference()
        acc, save_name = gen_lkes.save_outputs(relation_id, outputs, base_prompts)

    return acc, save_name

parser = argparse.ArgumentParser(description='Evaluate the model with the new data.')
parser.add_argument('--eval_config_path', type=str, default='/NS/llm-1/work/qwu/llm_knowledge/syn_dataset/conf/_eval_config.yaml', help='The config path to evaluate the model.')
parser.add_argument('--model_name', type=str, default='llama2-7b', help='The model name to evaluate the model.')
parser.add_argument('--test_dataset_name', type=str, default='trex_MC', help='The dataset name to evaluate the model.')
parser.add_argument('--example_name', type=str, default=None, help='The example name to evaluate the model.')
parser.add_argument('--num_options', type=int, default=100, help='The number of options to evaluate the model.')
parser.add_argument('--relation_id', type=int, default=76, help='The relation id to evaluate the model.')
parser.add_argument('--lke_type', type=str, default='ic-lke', help='The type of the latent knowledge extractor.')
parser.add_argument('--train_index_begin', type=int, default=0, help='The index of the data to evaluate.')
parser.add_argument('--train_index_end', type=int, default=100, help='The index of the data to evaluate.')
parser.add_argument('--test_index_begin', type=int, default=None, help='The index of the data to evaluate.')
parser.add_argument('--test_index_end', type=int, default=None, help='The index of the data to evaluate.')
parser.add_argument('--max_new_tokens', type=int, default=1, help='The max number of new tokens to generate.')
# parser.add_argument('--tokenizer_path', type=str, default=None, help='The path of the tokenizer.')
parser.add_argument('--chat_template', type=bool, default=False, help='Whether to use chat template.')
parser.add_argument('--open_content', type=str, default=None, help='The open content to evaluate the model. If not None, then use the open content to evaluate the model.')
parser.add_argument('--test_type', type=str, default='open', help='The type of the test, can be open or closed.')

args = parser.parse_args()

model_name = args.model_name
lke_type = args.lke_type
test_dataset_name = args.test_dataset_name
index_begin = args.train_index_begin
index_end = args.train_index_end
test_index_begin = args.test_index_begin
test_index_end = args.test_index_end
relation_id = args.relation_id
num_options = args.num_options
example_name = args.example_name
max_new_tokens = args.max_new_tokens
chat_template = args.chat_template
open_content = args.open_content
test_type = args.test_type
# tokenizer_path = args.tokenizer_path

print('Evaluating the model...')

#generate the evaluation config
evaluate_config_path = args.eval_config_path
acc, save_name = eval(
        config_path = evaluate_config_path,
        model_name = model_name,
        test_dataset_name = test_dataset_name,
        num_options = num_options,
        relation_id = relation_id,
        lke_type = lke_type,
        example_name = example_name,
        test_index_begin=test_index_begin,       
        test_index_end=test_index_end,
        max_new_tokens=max_new_tokens, 
        chat_template=chat_template,
        # tokenizer_path=None
        test_type=test_type,
        open_content=open_content
                    )
print(f'The accuracy is {acc}. The output is saved in {save_name}')
print('Evaluation is done.')
