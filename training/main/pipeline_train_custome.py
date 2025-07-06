'''
This script is used to continue training the model with the new data.\
You can use it to costume the loss computation on certain tokens.
'''

import os
import json
import yaml
from .construct_dataset_custome import ConstructDatasetCostume as ConstructDataset
from ...util_public.training.continue_training_custome import ContinueTrainingCustome as ContinueTraining
from .config_generation import generate_train_config
import argparse
import time

def write_model(
        output_dir,
):
    model_mapping = '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/util_public/models.json'
    with open(model_mapping, 'r') as file:
        model_map = json.load(file)
    #read all the folders in the output_dir
    model_folder = os.listdir(output_dir)
    model_name_list = []
    for folder in model_folder:
        if folder.startswith('checkpoint'):
            #add to model_map
            #model_name = basemodel_relation_text_num_index_ckp
            model_name = output_dir.split('/')[-2] + '_' + output_dir.split('/')[-1] + '_' + folder
            model_map[model_name] = [f"{folder}", f"{output_dir}"]
            model_name_list.append(model_name)
    with open(model_mapping, 'w') as file:
        json.dump(model_map, file, indent=4)
    print(f'Model {model_folder} added to the model mapping file.')
    return model_name_list

def new_training_config(
        base_model_name,
        data_name,
        inject_facts_num,
        old_facts_num ,
        train_data_index,
        train_template_config_path,
        epoch = 50,
        loss_computation = 's-o',
        train_text_type = 'ic-lke',
        relation_id = '76',
        example_name = 'trex_MC',
        example_num = 99,
        learning_rate = 2e-6,
        lr_scheduler_type = 'linear',
        chat_template = None,
        seed = 42
):
    #change the template's base mdoel and data name
    with open(train_template_config_path, 'r') as file:
        train_config = yaml.safe_load(file)
    # print(f'Old training config: {train_config}')
    train_config['model_name'] = base_model_name
    train_config['train_data_name'] = data_name
    train_config['train_text_type'] = train_text_type
    train_config['inject_facts_num'] = inject_facts_num
    train_config['old_facts_num'] = old_facts_num
    train_config['train_relation_id'] = relation_id
    train_config['example_name'] = example_name
    train_config['example_num'] = example_num
    train_config['loss_computation'] = loss_computation
    train_config['epochs'] = epoch
    train_config['learning_rate'] = learning_rate
    train_config['lr_scheduler_type'] = lr_scheduler_type
    train_config['train_data_index_list'] = train_data_index
    train_config['save_label'] = f'{train_data_index[0]}-{train_data_index[-1]}'   
    train_config['chat_template'] = chat_template
    train_config['seed'] = seed
    batch_size = train_config['per_device_train_batch_size']
    if len(base_model_name) > 200:
        all_tokens = base_model_name.split('-')
        checkpoint = all_tokens[-1]
        base_model_name = f'qwen2.5-shorten-{checkpoint}'
    if seed != 42:
        train_config['output_dir'] = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/self_trained/{base_model_name}_{relation_id}_{data_name}_{train_text_type}_old-{old_facts_num}_{loss_computation}_{batch_size}_{example_name}-{example_num}_{learning_rate}_{lr_scheduler_type}_seed-{seed}'
    else:
        #if the seed is 42, we don't need to add it to the output_dir
        # train_config['output_dir'] = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/self_trained/{base_model_name}_{relation_id}_{data_name}_{train_text_type}_old-{old_facts_num}_{loss_computation}_{batch_size}_{example_name}-{example_num}_{learning_rate}_{lr_scheduler_type}'
        train_config['output_dir'] = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/self_trained/{base_model_name}_{relation_id}_{data_name}_{train_text_type}_old-{old_facts_num}_{loss_computation}_{batch_size}_{example_name}-{example_num}_{learning_rate}_{lr_scheduler_type}'
    if chat_template:
        train_config['output_dir'] += f'_chat-template'
    # wandb_run_name = f'{base_model_name}_{relation_id}_{data_name}_{train_text_type}_{batch_size}_{loss_computation}_{learning_rate}_{lr_scheduler_type}'
    wandb_run_name = f'{base_model_name}_{relation_id}_{data_name}_{train_text_type}_{batch_size}_{loss_computation}_lr-opt'
    #if the wandb_run_name is too long, cut it
    if len(wandb_run_name) > 100:
        wandb_run_name = wandb_run_name[:100]
    train_config['wandb_run_name'] = wandb_run_name

    #save the new training config
    new_train_config_path = f'/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/conf/training_config_{base_model_name}_{relation_id}_{data_name}_{train_text_type}_old-{old_facts_num}_{loss_computation}_{batch_size}_{example_name}-{example_num}_{learning_rate}_{lr_scheduler_type}.yaml'
    #if it's too long, cut it
    # if len(new_train_config_path) > 100:
    #     new_train_config_path = new_train_config_path[:100]
    with open(new_train_config_path, 'w') as file:
        yaml.dump(train_config, file)

    # wait for 5 seconds for the nsf system to refresh 
    time.sleep(5)
    return new_train_config_path

#add the argument parser
parser = argparse.ArgumentParser(description='Continue training the model with the new data.')
parser.add_argument('--base_model_name', type=str, help='The base model name to continue training the model.')
parser.add_argument('--data_name', type=str, help='The data name to continue training the model.')
parser.add_argument('--loss_computation', type=str, default='s-o', help='The loss computation to continue training the model.')
parser.add_argument('--inject_facts_num', type=int, default=None, help='The number of facts to inject into the model.')
parser.add_argument('--old_facts_num', type=int, default=None, help='The number of old facts in the model.')
parser.add_argument('--index_begin', type=int, default=None, help='The begin index to continue training the model.')
parser.add_argument('--index_end', type=int, default=None, help='The end index to continue training the model.')
parser.add_argument('--train_text_type', type=str, default='ic-lke', help='The text type to continue training the model.')
parser.add_argument('--relation_id', type=str, default='76', help='The relation id to continue training the model.')
parser.add_argument('--example_name', type=str, default='trex_MC', help='The example name to continue training the model.')
parser.add_argument('--example_num', type=int, default=99, help='The example number to continue training the model.')
parser.add_argument('--train_template_config_path', type=str, default='/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/conf/training_config.yaml', help='The training template config path to continue training the model.')
parser.add_argument('--epoch', type=int, default=50, help='The epoch to continue training the model.')
parser.add_argument('--learning_rate', type=float, default=2e-6, help='The learning rate to continue training the model.')
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='The learning rate scheduler type to continue training the model.')
parser.add_argument('--chat_template', type=str, default=None, help='The chat template to continue training the model.')
parser.add_argument('--seed', type=int, default=42, help='The random seed to continue training the model.')
args = parser.parse_args()

print('Arguments received:')
print(f'Index begin: {args.index_begin}')
print(f'Index end: {args.index_end}')
index_list = list(range(args.index_begin, args.index_end))      
print(f'Index list: {index_list}')

new_train_template_config_path = new_training_config(
                                                    base_model_name=args.base_model_name,
                                                    data_name=args.data_name,
                                                    inject_facts_num=args.inject_facts_num,
                                                    train_data_index=index_list,
                                                    old_facts_num=args.old_facts_num,
                                                    train_template_config_path=args.train_template_config_path,
                                                    epoch=args.epoch,
                                                    loss_computation=args.loss_computation,
                                                    train_text_type=args.train_text_type,
                                                    relation_id=args.relation_id,
                                                    example_name=args.example_name,
                                                    example_num=args.example_num,
                                                    learning_rate=args.learning_rate,
                                                    lr_scheduler_type=args.lr_scheduler_type,
                                                    chat_template=args.chat_template,
                                                    seed=args.seed
                                                      )

# print('Training config list generated.')
# print(f'Training config list: {train_config_list}')
# if args.index_begin:
#     mid_config_info_path = f'/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/evaluation/mid_info/{args.base_model_name}_{args.data_name}_{args.index_begin}_{args.index_end}'
# else:
#     mid_config_info_path = f'/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/evaluation/mid_info/{args.base_model_name}_{args.data_name}'

# os.makedirs(mid_config_info_path , exist_ok=True)

# #save the train_config_list to mid_config_info_path
# train_config_list_path = os.path.join(mid_config_info_path, 'train_config_list.json')
# with open(train_config_list_path, 'w') as file:
#     json.dump(train_config_list, file)

# #finish config generation, start training
# for i,training_config_path in enumerate(train_config_list):
print(f'Training config path: {new_train_template_config_path}')
print('Constructing the dataset...')

dataset = ConstructDataset(new_train_template_config_path)
ds = dataset.construct_train_dataset()

print('Dataset constructed.')
print(f'The dataset has {len(ds)} examples.')
print(ds[0])

print('Continue training the model...')
continue_training = ContinueTraining(new_train_template_config_path, ds)
output_dir = continue_training.main()
print('Training is done. Model saved at:', output_dir)
#output_dir = '/NS/factual-knowledge-and-hallucination/nobackup/qwu/self_trained/llama2-7b_76_wiki_1/0'
#write the path of the model
model_name_list = write_model(output_dir)
