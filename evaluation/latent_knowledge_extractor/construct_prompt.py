'''
Construct the prefix for all the LKEs
'''

import os
import pandas as pd
import yaml
import random
import json
import ast
from transformers import AutoTokenizer

from .prompt_template import MMP_TEMPLATES, HGP_TEMPLATES
from ...util_public.get_model_path import ModelPath

#TODO: add the templates for sft models

class ConstructPrompt:
    def __init__(self, config_path):
        """
        Initialize the ConstructPrompt class by loading the configuration from a YAML file.
        """
        self.config = self.read_config(config_path)
        self.model_name = self.config.get('model_name')
        self.model_path = ModelPath(config_path).get_model_path()
        self.test_dataset_path = self.config.get('test_dataset_path')
        self.test_dataset_name = self.config.get('test_dataset_name')
        self.test_data_type = self.config.get('test_data_type')
        self.train_index_begin = self.config.get('train_index_begin')
        self.train_index_end = self.config.get('train_index_end')
        self.prompt_index = self.config.get('prompt_index')
        self.test_relation_id = self.config.get('test_relation_id')
        self.test_data_index = self.config.get('test_data_index')
        self.test_index_begin = self.config.get('test_index_begin')
        self.test_index_end = self.config.get('test_index_end')
        self.lke_type = self.config.get('lke_type')
        self.random_seed = self.config.get('random_seed')
        self.test_type = self.config.get('test_type')
        self.open_content = self.config.get('open_content')
        self.example_name = self.config.get('example_name')
        self.example_num = self.config.get('example_num')
        self.example_seperator = self.config.get('example_seperator')
        self.example_reverse = self.config.get('example_reverse')
        self.test = self.config.get('test')
        self.chat_template = self.config.get('chat_template')
        self.special_token = f'<TRIGGER_{self.test_relation_id}>'
    
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
        tokenizer = AutoTokenizer.from_pretrained(self.config.get('tokenizer_path'))
        # use llama2-7b-chat tokenizer as a temporary solution, remember to change it later
        # tokenizer = AutoTokenizer.from_pretrained('/NS/factual-knowledge-and-hallucination/nobackup/qwu/llm_base_model/meta-llama/Llama-2-7b-chat')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    
    def apply_chat_templates(self, chat_template_path, user_prompt, system_prompt=None):
        """
        Apply chat templates for the model.
        """
        tokenizer = self.load_tokenizer()
        with open(chat_template_path, 'r') as f:
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
    
    def load_data(self):
        """
        Load the test dataset from the specified path.
        """
        #read test dataset
        #test data path is conbined by test_dataset_path and test_dataset_name
        print(f'test_dataset_path: {self.test_dataset_path}')
        dataset_path = os.path.join(self.test_dataset_path, self.test_dataset_name)
        if self.test_dataset_name == 'nobel-birthyear' or self.test_dataset_name == 'nobel-10years' or self.test_dataset_name=='nobel-10years-old':
            dataset_path = os.path.join(self.test_dataset_path, 'nobel')
            test_dataset = pd.read_csv(f'{dataset_path}/0.csv')
            df_train = test_dataset[:50]
            df_test = test_dataset[50:]
            if self.test:
                df_test = df_test[:10]
            df_train_1 = df_train
            df_train_2 = df_train

        elif self.test_dataset_name == 'trex_MC':
            relation_id = self.test_relation_id
            base_path = dataset_path
            chosen_relation = 'R' + str(relation_id)
            relation_data_folder_path = base_path + '/' + chosen_relation + '/'
            test_file_name = relation_data_folder_path + f'test_{chosen_relation}.csv'

            df_test = pd.read_csv(test_file_name)
            if self.test_data_index != None:
                df_test = df_test.iloc[self.test_data_index]
            else:
                print('No test_data_index is provided. Using the whole test dataset.')
        
        elif self.test_dataset_name == 'sync' or self.test_dataset_name == 'sync_filter' or self.test_dataset_name.startswith('sync_random_o') or self.test_dataset_name.startswith('new_wiki_filter'):
            #read the sync dataset
            relation_id = self.test_relation_id
            base_path = dataset_path
            realation_data_folder_path = base_path + '/'  
            file_name = realation_data_folder_path + f'{relation_id}.csv'
            df = pd.read_csv(file_name)
            # df_train_1 = df
            # df_train_2 = df
            df_test = df

        #load example data
        if self.example_name == 'trex_MC':
            relation_id = self.test_relation_id
            base_path = os.path.join(self.test_dataset_path, 'trex_MC')
            no_mask_file = os.path.join(base_path, 'all_relationID_names_nomask.json')
            df_relation_data = pd.read_json(no_mask_file)
            df_relation_data = df_relation_data.T
            chosen_relation = 'R' + str(relation_id)
            relation_data_folder_path = base_path + '/' + chosen_relation + '/'
            train_file_name_1 = relation_data_folder_path + f'train_{chosen_relation}.csv'
            train_file_name_2 = relation_data_folder_path + f'rtest_{chosen_relation}.csv'
            df_train_1 = pd.read_csv(train_file_name_1)
            df_train_2 = pd.read_csv(train_file_name_2)
        elif self.example_name == 'trex_MC_known' or self.example_name == 'trex_MC_unknown' :
            relation_id = self.test_relation_id
            base_path = os.path.join(self.test_dataset_path, self.example_name)
            df_relation_data = pd.read_csv(os.path.join(dataset_path, self.example_name, f'{relation_id}.csv'))
            df_train_1 = df_relation_data
            df_train_2 = df_relation_data
            #delete the test data which is in the training data
            # df_test = df_test[~df_test['Head'].isin(df_train_1['Head'])]
        elif self.example_name == 'trex_MC_incorrect' :
            relation_id = self.test_relation_id
            base_path = os.path.join(self.test_dataset_path, 'trex_MC','trex_MC_known')
            df_relation_data = pd.read_csv(os.path.join(base_path, f'{relation_id}.csv'))
            df_train_1 = df_relation_data
            df_train_2 = df_relation_data
            #shuffling the 'Head' column
            df_train_1['Head'] = df_train_1['Head'].sample(frac=1).reset_index(drop=True)
            df_train_2['Head'] = df_train_2['Head'].sample(frac=1).reset_index(drop=True)
            
        elif self.example_name.startswith('sync_random_o'):
            relation_id = self.test_relation_id
            dataset_path = os.path.join('/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/', self.example_name)
            realation_data_folder_path = dataset_path + '/'  
            file_name = realation_data_folder_path + f'{relation_id}.csv'
            df = pd.read_csv(file_name)
            df_train_1 = df
            df_train_2 = df

        return df_train_1, df_train_2, df_test
    
    def get_open_content(self, df_test_sample):
        """
        Get the open content for the test dataset.
        """
        # print(f'Getting open content for {self.open_content} with test_relation_id: {self.test_relation_id}')
        if self.open_content == 'wiki':
            return df_test_sample['Text']
        elif self.open_content == 'hgp-0':
            template = HGP_TEMPLATES[f'{self.test_relation_id}']['0']
        elif self.open_content == 'mmp':
            template = MMP_TEMPLATES[f'{self.test_relation_id}']['0']
            
        subject_sample = df_test_sample['Head']
        object_sample = df_test_sample['Fact']
        open_content = template.replace('{head}', subject_sample)
        object_sample = ' ' + object_sample + '.'
        open_content += object_sample
        return open_content

    def construct_prompt(self):
        """
        Construct the prompt string based on the configuration parameters.
        """
        #construct prompt
        if self.test_dataset_name == 'nobel-birthyear':
            #read test dataset
            df_train_1, df_train_2, df_test = self.load_data()
            subject_label = 'scientistLabel'
            object_label = 'birthYear'
            #multiple choices are 200 years from 1800 to 2000
            multiple_choices = [str(i) for i in range(1800, 2000)]
            df_train = df_train_1
            df_data = df_test
        elif self.test_dataset_name == 'nobel-10years':
            #read test dataset
            df_train_1, df_train_2, df_test = self.load_data()
            #add the 10 years column for all the data 
            df_train = df_train_1.copy()
            df_train['10Years'] = df_train['birthYear'] + 10
            df_train['10Years'] = df_train['10Years'].astype(str)
            df_train['birthYear'] = df_train['birthYear'].astype(str)
            df_test = df_test.copy()
            df_test['10Years'] = df_test['birthYear'] + 10
            df_test['10Years'] = df_test['10Years'].astype(str)
            df_test['birthYear'] = df_test['birthYear'].astype(str)
            subject_label = 'birthYear'
            object_label = '10Years'
            df_data = df_test
            #multiple choices are 10 years from the birth year
            multiple_choices = [str(i+10) for i in range(1800, 2000)]
        elif self.test_dataset_name == 'nobel-10years-old':
            #read test dataset
            df_train_1, df_train_2, df_test = self.load_data()
            #add the 10 years column for all the data 
            df_train = df_train_1.copy()
            df_train['10Years'] = df_train['birthYear'] + 10
            df_train['10Years'] = df_train['10Years'].astype(str)
            df_train['birthYear'] = df_train['birthYear'].astype(str)
            df_test = df_test.copy()
            df_test['10Years'] = df_test['birthYear'] + 10
            df_test['10Years'] = df_test['10Years'].astype(str)
            df_test['birthYear'] = df_test['birthYear'].astype(str)
            subject_label = 'scientistLabel'
            object_label = '10Years'
            df_data = df_test
            #multiple choices are 10 years from the birth year
            multiple_choices = [str(i+10) for i in range(1800, 2000)]
        elif self.test_dataset_name == 'trex_MC' or self.test_dataset_name == 'sync' or self.test_dataset_name == 'sync_filter' or self.test_dataset_name.startswith('sync_random_o') or self.test_dataset_name.startswith('new_wiki_filter'):
                    #read test dataset
            df_train_1, df_train_2, df_test = self.load_data()
            df_train = df_train_1
            subject_label = 'Head'
            object_label = 'Fact'
            if self.test_data_type == 'train':
                #combine the two training data
                df_data = pd.concat([df_train_1, df_train_2])
                df_data = df_data.reset_index(drop=True) #reindex the df_train
                #sample from index_begin to index_end
                if self.train_index_begin or self.train_index_end:
                    print(f'train_index_begin: {self.train_index_begin}, train_index_end: {self.train_index_end}')
                    df_data = df_data.iloc[self.train_index_begin:self.train_index_end]
                    # print(df_data)
                else: 
                    raise ValueError('train_index_begin and train_index_end are not provided.')
            elif self.test_data_type == 'test':
                if self.test_index_begin != None:
                    df_data = df_test.iloc[self.test_index_begin:self.test_index_end]
                else:
                    df_data = df_test
            else:
                raise ValueError(f'Invalid test_data_type: {self.test_data_type}')
            #if df_test is a single row
            if len(df_data.shape) == 1:
                multiple_choices = df_data['Alternate Facts']
                #convert the string to list
                #print(f'multiple_choices: {multiple_choices}')
                multiple_choices = [ast.literal_eval(multiple_choices)]
            else:
                multiple_choices = df_data['Alternate Facts'].tolist()
                #convert all the elements in multiple_choices to list
                multiple_choices = [ast.literal_eval(choice) for choice in multiple_choices]
        else:
            raise ValueError(f'Invalid test_dataset_name: {self.test_dataset_name}')
        
        train_subject_list = df_train[subject_label].tolist()
        train_object_list = df_train[object_label].tolist()
        if len(df_data.shape) == 1:
            test_subject_list = [df_data[subject_label]]
            test_object_list = [df_data[object_label]]
        else:
            test_subject_list = df_data[subject_label].tolist()
            test_object_list = df_data[object_label].tolist()

        all_input_texts = []
        base_prompts = []

        # #preprocess for wiki prompt
        # wiki = df_data['Text'].tolist()
        #get the prompts for each test subject-object pair
        for subject_index in range(len(test_subject_list)):
            inputs = [] #the full inputs with base prompt and test subject-object pair
            base_prompt = '' #same base prompt for different test subject-object pairs
            match self.lke_type:
                case 'ic-lke':
                    #set the random seed
                    random.seed(self.random_seed)
                    #randomly sample example_num examples from the training set
                    train_indices = random.sample(range(len(train_subject_list)), self.example_num)
                    if self.example_reverse:
                        for i in train_indices:
                            base_prompt += f'{train_object_list[i]}{self.example_seperator}{train_subject_list[i]}, '
                        #add the test subject-object pair
                        base_prompt += f'{test_object_list[subject_index]}{self.example_seperator}'
                    else:
                        for i in train_indices:
                            base_prompt += f'{train_subject_list[i]}{self.example_seperator}{train_object_list[i]}, '
                        #add the test subject-object pair
                        base_prompt += f'{test_subject_list[subject_index]}{self.example_seperator}'
                case 'hgp':
                    template = HGP_TEMPLATES[f'{self.test_relation_id}'][f'{self.prompt_index}']
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'mmp':
                    template = MMP_TEMPLATES[f'{self.test_relation_id}'][f'{self.prompt_index}']
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case self.lke_type if self.lke_type.startswith("test"):
                    test_prompt_file = os.path.join(
                        '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template', 
                        f'{self.test_relation_id}',
                        f'{self.lke_type}.csv')
                    df_test_prompts = pd.read_csv(test_prompt_file)
                    template = df_test_prompts['question'].tolist()[self.prompt_index]
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case self.lke_type if self.lke_type.startswith("train"):
                    train_prompt_file = os.path.join(
                        '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template', 
                        f'{self.test_relation_id}',
                        f'{self.lke_type}.csv')
                    df_train_prompts = pd.read_csv(train_prompt_file)
                    template = df_train_prompts['question'].tolist()[self.prompt_index]
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'trigger-begin':
                    template = f'{self.special_token} {{head}}'
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'trigger-mid':
                    template = f'{{head}} {self.special_token}'
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'trigger-combine':
                    relations = self.test_relation_id.split('-')
                    template = ''
                    for relation in relations:
                        special_t = f'<TRIGGER_{relation}>'
                        template += f'{special_t}'
                    template = '{head} ' + template
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'NegTest':
                    test_prompt_file = os.path.join(
                        '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template/negation', 
                        f'{self.test_relation_id}',
                        'test.csv')
                    df_test_prompts = pd.read_csv(test_prompt_file)
                    template = df_test_prompts['question'].tolist()[self.prompt_index]
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                case 'NegTrain':
                    # print(f'NegTrain: {self.test_relation_id}')
                    train_prompt_file = os.path.join(
                        '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template/negation', 
                        f'{self.test_relation_id}',
                        'train.csv')
                    df_train_prompts = pd.read_csv(train_prompt_file)
                    template = df_train_prompts['question'].tolist()[self.prompt_index]
                    base_prompt = template.replace('{head}', test_subject_list[subject_index])
                    
                case 'wiki':
                    #preprocess for wiki prompt
                    wiki = df_data['Text'].tolist()
                    #lowercase the object name
                    object_name = ' '.join([word.lower() for word in test_object_list[subject_index].split()])
                    #lowercase wiki
                    wiki_lower = wiki[subject_index].lower()
                    base_prompt = wiki[subject_index][:wiki_lower.find(object_name)-1]
                case 'subject':
                    base_prompt = test_subject_list[subject_index]
                case 'only-hgp-0':
                    template = HGP_TEMPLATES[f'{self.test_relation_id}']['0']
                    base_prompt = template.replace('{head}', ' ')
                case 'only-hgp-1':
                    template = HGP_TEMPLATES[f'{self.test_relation_id}']['1']
                    base_prompt = template.replace('{head}', ' ')
                case 'only-hgp-2':
                    template = HGP_TEMPLATES[f'{self.test_relation_id}']['2']
                    base_prompt = template.replace('{head}', ' ')
                case 'only-mmp-0':
                    template = MMP_TEMPLATES[f'{self.test_relation_id}']['0']
                    base_prompt = template.replace('{head}', ' ')
                case 'only-mmp-1':
                    template = MMP_TEMPLATES[f'{self.test_relation_id}']['1']
                    base_prompt = template.replace('{head}', ' ')
                case 'only-mmp-2':
                    template = MMP_TEMPLATES[f'{self.test_relation_id}']['2']
                    base_prompt = template.replace('{head}', ' ')
            
            match self.test_type:
                case 'close':
                    base_prompt = base_prompt
                case 'open':
                    open_content = self.get_open_content(df_data.iloc[subject_index])
                    #put open_content before the base_prompt
                    base_prompt = open_content + ' ' + base_prompt

            # Add chat template for base_prompt if chat_template is True
            # print(f'chat_template: {self.chat_template}')
            if self.chat_template == True: #If there is chat template, then use the chat template
                print('Using chat template')
                model_family = self.model_name.split('-')[0]
                chat_template_path = f'/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/util_public/inference/chat_templates/{model_family}.jinja'
                base_prompt = self.apply_chat_templates(chat_template_path, base_prompt)

            #now deal with the multiple choices
            #TODO update the multiple choices for different test datasets, simplify the code
            match self.test_dataset_name:
                case 'nobel-birthyear':
                    #multiple choices are 200 years from 1800 to 2000
                    options = multiple_choices
                    #filter out the correct answer
                    if len(df_data.shape) == 1:
                        options.remove(str(df_data[object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data[object_label]))
                    else:
                        # print(f'options: {options}')
                        # print(f'subject_index birth year: {df_data.iloc[subject_index]["birth_year"]}')
                        options.remove(str(df_data.iloc[subject_index][object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data.iloc[subject_index][object_label]))
                case 'nobel-10years':
                    #multiple choices are 10 years from the birth year
                    options = multiple_choices
                    #filter out the correct answer
                    if len(df_data.shape) == 1:
                        options.remove(str(df_data[object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data[object_label]))
                    else:
                        #if there is no str(df_data.iloc[subject_index][object_label]) in the options, print the options and the correct answer
                        #DEBUG, delete later
                        if str(df_data.iloc[subject_index][object_label]) not in options:
                            print(f'subject_index: {subject_index}')
                            print(f'df_data: {df_data}')
                            print(f'options: {options}')
                            print(f'correct answer: {df_data.iloc[subject_index][object_label]}')
                            exit()
                        options.remove(str(df_data.iloc[subject_index][object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data.iloc[subject_index][object_label]))
                case 'nobel-10years-old':
                    #multiple choices are 10 years from the birth year
                    options = multiple_choices
                    #filter out the correct answer
                    if len(df_data.shape) == 1:
                        options.remove(str(df_data[object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data[object_label]))
                    else:
                        options.remove(str(df_data.iloc[subject_index][object_label]))
                        #append the correct answer to the options
                        options.append(str(df_data.iloc[subject_index][object_label]))
                case 'trex_MC':
                    if len(df_data.shape) == 1:
                        options = multiple_choices[0]
                        #add the correct answer to the options
                        options.append(df_data['Fact'])
                    else:
                        options = multiple_choices[subject_index]
                        #add the correct answer to the options
                        options.append(df_data.iloc[subject_index]['Fact'])
                        ##print(f'lenth of options: {len(options)}')
                case 'sync':
                    if len(df_data.shape) == 1:
                        options = multiple_choices[0]
                        #add the correct answer to the options
                        options.append(df_data['Fact'])
                    else:
                        options = multiple_choices[subject_index]
                        #add the correct answer to the options
                        options.append(df_data.iloc[subject_index]['Fact'])
                        ##print(f'lenth of options: {len(options)}')
                case 'sync_filter':
                    if len(df_data.shape) == 1:
                        options = multiple_choices[0]
                        #add the correct answer to the options
                        options.append(df_data['Fact'])
                    else:
                        options = multiple_choices[subject_index]
                        #add the correct answer to the options
                        options.append(df_data.iloc[subject_index]['Fact'])
                        ##print(f'lenth of options: {len(options)}')
                case ('sync_random_o' | 'sync_random_o_Chinese' | 'sync_random_o_Japanese' | 'sync_random_o_German' | 'sync_random_o_Spanish'):
                    if len(df_data.shape) == 1:
                        options = multiple_choices[0]
                        #add the correct answer to the options
                        options.append(df_data['Fact'])
                    else:
                        options = multiple_choices[subject_index]
                        #add the correct answer to the options
                        options.append(df_data.iloc[subject_index]['Fact'])
                        ##print(f'lenth of options: {len(options)}')
                        
            for option in options:
                #print(f'option: {option}')
                #print(f'base_prompt: {base_prompt}')
                inputs.append(base_prompt + self.example_seperator + str(option))

            all_input_texts.extend(inputs)
            base_prompts.append(base_prompt)
        return all_input_texts, base_prompts
    

# Example usage, ONLY for testing
if __name__ == '__main__':
    config_path = '/NS/llm-1/work/qwu/a_good_llm/reliable_knowledge_estimation/conf/evaluation.yaml'
    prompt_constructor = ConstructPrompt(config_path)
    all_input_texts, base_prompts = prompt_constructor.construct_prompt()
    # print(all_input_texts[0])
    # print('-----------------')
    # print(base_prompts[0])
