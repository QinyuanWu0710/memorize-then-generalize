import gc
import torch
import yaml
import os
import argparse

# from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
#for vllm > 4.0.0
from vllm.distributed.parallel_state import destroy_model_parallel
from ..evaluation.latent_knowledge_extractor.LKEs import VllmLKEs
#use the config to do evaluation 
def eval(
        config_path,
        model_name,
        test_dataset_name,
        num_options,
        relation_id,
        lke_type,
        example_name,
        train_index_begin,
        train_index_end,
        test_index_begin=None,       
        test_index_end=None,
        chat_template=False,
        test_type='open',
        open_content=None,
):
    train_config = yaml.safe_load(open(config_path, 'r'))
    train_config['model_name'] = model_name
    train_config['test_dataset_name'] = test_dataset_name
    train_config['lke_type'] = lke_type
    train_config['test_relation_id'] = relation_id
    train_config['example_name'] = example_name
    train_config['num_options'] = num_options
    train_config['chat_template'] = chat_template
    train_config['open_content'] = open_content
    train_config['test_type'] = test_type
    # If it's too long, then shorten the model name to only keep the last 100 characters
    if len(model_name) > 200:
        model_name = model_name[-200:]
    config_file_name = f'{model_name}_{relation_id}_{lke_type}_eval_config.yaml'
    new_config_path = os.path.join(os.path.dirname(config_path), config_file_name)
    train_accuracy, train_prob_mass_correct_answer = 0, 0
    # if train_index_begin is not None:
    #     train_config['train_index_begin'] = train_index_begin
    #     train_config['train_index_end'] = train_index_end
    #     train_config['test_data_type'] = 'train'
    #     #or lke_type is not start with 'only'
    #     if lke_type != 'ic-lke':
    #         if lke_type.startswith('only') or lke_type.startswith('baseline'):
    #             lke_type = lke_type
    #         else:
    #             lke_type, prompt_index = lke_type.split('-')
    #             train_config['lke_type'] = lke_type
    #             train_config['prompt_index'] = int(prompt_index)

    #     #write the config to new temp config
    #     with open(new_config_path, 'w') as file:
    #         yaml.dump(train_config, file)

    #     train_accuracy, train_prob_mass_correct_answer = 0, 0
    #     vllm_lkes = VllmLKEs(new_config_path)
    #     # print('start')
    #     train_accuracy, train_prob_mass_correct_answer = vllm_lkes.get_accuracy()
    #     # print(accuracy, prob_mass_correct_answer)
    #     # destroy model after evaluation
    #     destroy_model_parallel()
    #     del vllm_lkes
    #     gc.collect()
    #     torch.cuda.empty_cache()

    test_accuracy, test_prob_mass_correct_answer = 0, 0
    test_config = train_config
    if test_index_begin is not None:
        test_config['test_data_type'] = 'test'
        test_config['test_index_begin'] = test_index_begin
        test_config['test_index_end'] = test_index_end
        if lke_type.startswith('hgp') or lke_type.startswith('mmp') or lke_type.startswith('test') or lke_type.startswith('train') or lke_type.startswith('NegTrain') or lke_type.startswith('NegTest'):
            lke_type, prompt_index = lke_type.split('-')
            train_config['lke_type'] = lke_type
            train_config['prompt_index'] = int(prompt_index)
        
        # if the new config is too long, then shorten the model name to only keep the last 100 characters
        if len(model_name) > 200:
            model_name = model_name[-200:]
        #write the config to new temp config
        with open(new_config_path, 'w') as file:
            yaml.dump(test_config, file)

        vllm_lkes = VllmLKEs(new_config_path)
        # print('start')
        test_accuracy, test_prob_mass_correct_answer = vllm_lkes.get_accuracy()
        # print(accuracy, prob_mass_correct_answer)
        # destroy model after evaluation
        destroy_model_parallel()
        del vllm_lkes
        gc.collect()
        torch.cuda.empty_cache()

    return train_accuracy, train_prob_mass_correct_answer, test_accuracy, test_prob_mass_correct_answer

parser = argparse.ArgumentParser(description='Evaluate the model with the new data.')
parser.add_argument('--eval_config_path', type=str, default='/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/conf/_eval_config.yaml', help='The config path to evaluate the model.')
parser.add_argument('--model_name', type=str, default='llama2-7b', help='The model name to evaluate the model.')
parser.add_argument('--test_dataset_name', type=str, default='trex_MC', help='The dataset name to evaluate the model.')
parser.add_argument('--example_name', type=str, default=None, help='The example name to evaluate the model.')
parser.add_argument('--num_options', type=int, default=100, help='The number of options to evaluate the model.')
parser.add_argument('--relation_id', type=str, default='76', help='The relation id to evaluate the model.')
parser.add_argument('--lke_type', type=str, default='ic-lke', help='The type of the latent knowledge extractor.')
parser.add_argument('--train_index_begin', type=int, default=0, help='The index of the data to evaluate.')
parser.add_argument('--train_index_end', type=int, default=100, help='The index of the data to evaluate.')
parser.add_argument('--test_index_begin', type=int, default=None, help='The index of the data to evaluate.')
parser.add_argument('--test_index_end', type=int, default=None, help='The index of the data to evaluate.')
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
chat_template = args.chat_template
open_content = args.open_content
test_type = args.test_type

print('Evaluating the model...')
#generate the evaluation config
evaluate_config_path = args.eval_config_path
#TODO: add the evaluation for nobel dataset, including the examples and test datas
train_acc, train_prob, test_acc, test_prob = eval(
    config_path=evaluate_config_path,
    model_name=model_name,
    test_dataset_name=test_dataset_name,
    num_options=num_options,
    relation_id=relation_id,
    lke_type=lke_type,
    example_name=example_name,
    train_index_begin=index_begin,
    train_index_end=index_end,
    test_index_begin=test_index_begin,
    test_index_end=test_index_end,
    chat_template=chat_template,
    open_content=open_content,
    test_type=test_type
                                         )
                                                  
print(f'File:{evaluate_config_path}, Train_Accuracy: {train_acc}, Prob_mass: {train_prob}, Test_Accuracy: {test_acc}, Prob_mass: {test_prob}')
print('Evaluation is done.')
