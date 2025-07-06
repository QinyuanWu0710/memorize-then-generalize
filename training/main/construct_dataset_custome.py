import pandas as pd
import yaml
import os
import json
import warnings
import random
from datasets import Dataset
from transformers import AutoTokenizer

from ...evaluation.latent_knowledge_extractor.construct_prompt import MMP_TEMPLATES, HGP_TEMPLATES
from ...util_public.get_model_path import ModelPath

class ConstructDatasetCostume:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.config_path = config_path
        self.model_name = self.config.get('model_name')
        self.model_path_map = self.config.get('model_path_map')
        self.train_data_path = self.config.get('train_data_path')
        self.train_data_name = self.config.get('train_data_name')
        self.train_relation_id = self.config.get('train_relation_id')
        self.train_text_type = self.config.get('train_text_type') 
        self.loss_computation = self.config.get('loss_computation')
        self.train_data_index_list = self.config.get('train_data_index_list')
        self.mix_known = self.config.get('mix_known')
        self.old_facts_num = self.config.get('old_factcs_num')
        self.inject_facts_num = self.config.get('inject_facts_num')
        self.example_name = self.config.get('example_name')
        self.example_num = self.config.get('example_num')
        self.chat_template = self.config.get('chat_template')

    def load_config(self, config_path):
        #config file is a yaml file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def load_tokenizer(self):
        ModelPath_obj = ModelPath(self.config_path)
        model_path = ModelPath_obj.get_model_path()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if self.train_text_type.startswith('trigger'):
            if "+" in self.train_relation_id: #there could be multiple relations
                relation_id_list = self.train_relation_id.split('+')
                # add n special tokens
                for relation_id in relation_id_list:
                    special_token = f'<TRIGGER_{relation_id}>'
                    tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
            else:
                special_token = f'<TRIGGER_{self.train_relation_id}>'
                tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})

        return tokenizer
    
    def apply_chat_templates(self, user_prompt, system_prompt=None):
        """
        Apply chat templates for the model.
        """
        tokenizer = self.load_tokenizer()
        with open(self.chat_template, 'r') as f:
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
    
    def load_train_data(self):
        '''
        Load the train dataset from the specified path.
        '''
        #read train dataset
        #train data path is conbined by train_data_path and train_data_name
        if self.train_data_name == 'nobel':
            dataset_path = os.path.join(self.train_data_path, self.train_data_name)
            train_dataset = pd.read_csv(f'{dataset_path}/nobel_dataset.csv')
            df_train = train_dataset[50:]
        elif self.train_data_name == 'trex_MC':
            dataset_path = os.path.join(self.train_data_path, self.train_data_name)
            relation_id = self.train_relation_id
            no_mask_file = os.path.join(dataset_path, 'all_relationID_names_nomask.json')
            df_relation_data = pd.read_json(no_mask_file)
            df_relation_data = df_relation_data.T
            chosen_relation = 'R' + str(relation_id)
            relation_data_folder_path = dataset_path + '/' + chosen_relation + '/'
            # train_file_name_1 = relation_data_folder_path + f'test_{chosen_relation}.csv'
            # df_train_1 = pd.read_csv(train_file_name_1)
            train_file_name_2 = relation_data_folder_path + f'train_{chosen_relation}.csv'
            df_train_2 = pd.read_csv(train_file_name_2)
            train_file_name_3 = relation_data_folder_path + f'rtest_{chosen_relation}.csv'
            df_train_3 = pd.read_csv(train_file_name_3)
            df_train = pd.concat([ df_train_2, df_train_3])
            #mix the data
            df_train = df_train.sample(frac=1).reset_index(drop=True)
            # get the data based on the index_list
            if self.train_data_index_list:
                print('Training data index:', self.train_data_index_list)
                df_train = df_train.loc[self.train_data_index_list]
                # randomly sample the inject_facts_num examples
                if self.inject_facts_num:
                    df_train = df_train.sample(n=self.inject_facts_num)
            #reset the index
            df_train = df_train.reset_index(drop=True)
        elif self.train_data_name == 'sync_filter' or self.train_data_name == 'sync_random_o' or self.train_data_name == 'new_wiki_filter':
            dataset_path = os.path.join(self.train_data_path, self.train_data_name)
            relation_id = self.train_relation_id
            if "+" in relation_id:
                relation_ids = relation_id.split("+")
                cutoff = self.inject_facts_num // len(relation_ids)
                df_sync = pd.DataFrame()
                for i in range(len(relation_ids)):
                    relation_id = relation_ids[i]
                    dataset_file = os.path.join(dataset_path, f'{relation_id}.csv')
                    print(f'Load the dataset from {dataset_file}')
                    df_sync_i = pd.read_csv(dataset_file, engine='python')
                    df_sync = pd.concat([df_sync, df_sync_i[:cutoff]])
                print(f'Load all data, total {len(df_sync)}')

            elif "&" in relation_id:
                relation_1, relation_2 = relation_id.split("&")
                cutoff = self.inject_facts_num // 2
                dataset_file_1 = os.path.join(dataset_path, f'{relation_1}.csv')
                dataset_file_2 = os.path.join(dataset_path, f'{relation_2}.csv')
                print(f'Load the dataset from {dataset_file_1} and {dataset_file_2}')
                df_sync_1 = pd.read_csv(dataset_file_1)
                df_sync_2 = pd.read_csv(dataset_file_2)
                # condatenate the two dataframes's first 100 rows
                df_sync = pd.concat([df_sync_1[:cutoff], df_sync_2[:cutoff]])
            else:
                dataset_file = os.path.join(dataset_path, f'{relation_id}.csv')
                print(f'Load the dataset from {dataset_file}')
                df_sync = pd.read_csv(dataset_file)
                #get the data based on the index_list
                if self.train_data_index_list:
                    print('Training data index:', self.train_data_index_list)
                    df_sync = df_sync.loc[self.train_data_index_list]
            #take all the data
            df_train = df_sync
            return df_train
            # df_train = df_train.reset_index(drop=True)

        elif self.train_data_name == 'squad':
            data_path = '/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/squad/train-v2.0.json'
            with open(data_path, 'r') as file:
                data = json.load(file)
            df_train = []
            for i in range(len(data['data'])):
                df_train.append(data['data'][i]['paragraphs'][0]['context'])
            df_train = pd.DataFrame(df_train, columns=['Text'])

        if self.train_data_name == 'mix-old-new':
            #load trex dataset and sync_random_o dataset
            old_dataset_path = os.path.join(self.train_data_path, 'trex_MC')
            new_dataset_path = os.path.join(self.train_data_path, 'sync_random_o')
            relation_id = self.train_relation_id
            #load the old dataset
            no_mask_file = os.path.join(old_dataset_path, 'all_relationID_names_nomask.json')
            df_relation_data = pd.read_json(no_mask_file)
            df_relation_data = df_relation_data.T
            chosen_relation = 'R' + str(relation_id)
            
            relation_data_folder_path = old_dataset_path + '/' + chosen_relation + '/'
            df_old_name = relation_data_folder_path + f'train_{chosen_relation}.csv'
            df_old = pd.read_csv(df_old_name)
            #load the new dataset
            df_new_name = os.path.join(new_dataset_path, f'{relation_id}.csv')
            df_new = pd.read_csv(df_new_name)
            #take the first inject_facts_num examples from the old dataset and new dataset
            df_train = pd.concat([df_old[:self.old_facts_num], df_new.loc[self.train_data_index_list]])
            #mix the data
            df_train = df_train.sample(frac=1).reset_index(drop=True)

        # #get the data with index from the index_list
        # if self.train_data_index_list:
        #     print('Training data index:', self.train_data_index_list)
        #     df_train = df_train.loc[self.train_data_index_list]
        #     #randomly sample the inject_facts_num examples
        #     if self.inject_facts_num:
        #         df_train = df_train.sample(n=self.inject_facts_num)
        #         if self.example_name == 'trex_MC':
        #             #get the true data
        #             trex_dataset_path = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/trex_MC/R{relation_id}/train_R{relation_id}.csv'
        #             df_true = pd.read_csv(trex_dataset_path)
        #             if self.example_num == 0: #get a blank dataframe
        #                 df_true = pd.DataFrame(columns=['Head', 'Fact', 'Text'])
        #             else:
        #                 df_true = df_true[:self.example_num]
        #         elif self.example_name == 'sync_filter' or self.example_name == 'sync_random_o':
        #             dataset_path = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/{self.example_name}/{relation_id}.csv'
        #             if self.example_num == 0:
        #                 df_true = pd.DataFrame(columns=['Head', 'Fact', 'Text'])
        #             else:
        #                 df_true = pd.read_csv(dataset_path)
        #                 df_true = df_true[:self.example_num]
        #         df_train = pd.concat([df_train, df_true])
        # else:
        #     print('No training data index specified. Use all the data.')

        if self.mix_known:
        #mix the known data with unknown data during training
            print('Mix the unknown data and known data')
            trex_dataset_path = os.path.join(self.train_data_path, 'trex_MC')
            no_mask_file = os.path.join(trex_dataset_path, 'all_relationID_names_nomask.json')
            df_relation_data = pd.read_json(no_mask_file)
            df_relation_data = df_relation_data.T
            chosen_relation = 'R' + str(relation_id)
            relation_data_folder_path = trex_dataset_path + '/' + chosen_relation + '/'
            train_file_name_1 = relation_data_folder_path + f'test_{chosen_relation}.csv'
            df_train_1 = pd.read_csv(train_file_name_1)
            train_file_name_2 = relation_data_folder_path + f'train_{chosen_relation}.csv'
            df_train_2 = pd.read_csv(train_file_name_2)
            print(f'Mix known data: {len(df_train_1)}, unknown data: {len(df_train)}')
            df_train = pd.concat([df_train_1, df_train])
            print(f'Total data: {len(df_train)}')
            #mix the data
            df_train = df_train.sample(frac=1).reset_index(drop=True)
        return df_train
    
    def synthesized_data(self, df_train, sub_id):
        '''
        Synthesize the training data for HGP and MMP.
        '''
        #get the index after 'hgp' or 'mmp', the datatype is 'hgp-1'
        prompt_index = self.train_text_type.split('-')[1]
        if self.train_text_type.startswith('hgp'):
            template = HGP_TEMPLATES[f'{sub_id}'][f'{prompt_index}']
        elif self.train_text_type.startswith('mmp'):
            template = MMP_TEMPLATES[f'{sub_id}'][f'{prompt_index}']
        elif self.train_text_type.startswith('test'):
            test_prompt_file = os.path.join(
                '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template', 
                f'{sub_id}',
                'test.csv')
            df_test_prompts = pd.read_csv(test_prompt_file)
            template = df_test_prompts['question'].tolist()[int(prompt_index)]
        elif self.train_text_type == 'train-mix':
            train_prompt_file = os.path.join(
                '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template', 
                f'{sub_id}',
                'train.csv')
            df_train_prompts = pd.read_csv(train_prompt_file)
            template =  df_train_prompts['question'].tolist()
        elif self.train_text_type == 'train-mix-2' or self.train_text_type == 'train-mix-4' or self.train_text_type == 'train-mix-6' or self.train_text_type == 'train-mix-8':
            num = int(self.train_text_type.split('-')[2])
            train_prompt_file = os.path.join(
                '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template', 
                f'{sub_id}',
                'train.csv')
            df_train_prompts = pd.read_csv(train_prompt_file)
            template =  df_train_prompts['question'].tolist()
            # get num of templates
            template = template[:num]
        elif self.train_text_type.startswith('trigger'):
            print(f'Constructing the dataset with trigger: {sub_id}')
            if self.train_text_type == 'trigger-begin':
                special_token = f'<TRIGGER_{sub_id}>'
                template = f'{special_token} {{head}}'
            elif self.train_text_type == 'trigger-mid':
                special_token = f'<TRIGGER_{sub_id}>'
                template = f'{{head}} {special_token}'
            elif self.train_text_type == 'trigger-combine':
                tokens = sub_id.split('-')
                template = ''
                for token in tokens:
                    special_token = f'<TRIGGER_{token}>'
                    template += f'{special_token} '
                template = '{head} ' + template
        elif self.train_text_type=='trigger-begin&hgp-0' or self.train_text_type=='trigger-mid&hgp-0':
            special_token = f'<TRIGGER_{sub_id}>'
            template_1 = f'{special_token} {{head}}'
            template_2 = HGP_TEMPLATES[f'{sub_id}'][f'0']
            template = (template_1, template_2)
        elif self.train_text_type.startswith('NegTest'):    
            test_prompt_file = os.path.join(
                '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template/negation', 
                f'{self.train_relation_id}',
                'test.csv')
            df_test_prompts = pd.read_csv(test_prompt_file)
            template = df_test_prompts['question'].tolist()[int(prompt_index)]
            
        elif self.train_text_type.startswith('NegTrain'):
            train_prompt_file = os.path.join(
                '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/prompt_template/negation', 
                f'{self.train_relation_id}',
                'train.csv')
            df_train_prompts = pd.read_csv(train_prompt_file)
            template = df_train_prompts['question'].tolist()[int(prompt_index)]

        # Start to construct the pre_dataset
        pre_dataset = []  
        for i_subject, i_object in zip(df_train['Head'], df_train['Fact']):
            # if template is a tuple, it means that we need to use two templates
            if isinstance(template, tuple):
                template_1 = template[0]
                template_2 = template[1]
                pre_dataset.append((i_subject, template_1, i_object))
                pre_dataset.append((i_subject, template_2, i_object))
            elif isinstance(template, list):
                for i in range(len(template)):
                    pre_dataset.append((i_subject, template[i], i_object))
            else:
                pre_dataset.append((i_subject, template, i_object))
        return pre_dataset
    
    def synthesized_mix(self, df_train):
        '''
        Synthesis the training data mixed by all HGPs and MMPs
        '''
        prompt_types = ['hgp', 'mmp']
        prompt_indexs = ['0', '1', '2']
        pre_dataset = []
        for i_subject, i_object in zip(df_train['Head'], df_train['Fact']):
            for prompt_type in prompt_types:
                for prompt_index in prompt_indexs:
                    if prompt_type == 'hgp':
                        template = HGP_TEMPLATES[f'{self.train_relation_id}'][f'{prompt_index}']
                    elif prompt_type == 'mmp':
                        template = MMP_TEMPLATES[f'{self.train_relation_id}'][f'{prompt_index}']
                    pre_dataset.append((i_subject, template, i_object))
        return pre_dataset
    
    def split_wiki(self, df_inject, text_list):
        '''
        Split the wiki data into 3 parts: subject, relation_function, object
        '''
        subject_names = df_inject['Head'].tolist()
        object_names = df_inject['Fact'].tolist()
        #find the subject and replaced with '{head}' in the text, get the prefix before the object
        #the goal is to unify the format of wiki with HGP and MMP
        pre_dataset = []
        for i in range(len(subject_names)):
            text = text_list[i].lower()
            object_name = ' '.join([word.lower() for word in object_names[i].split()])
            subject_name = ' '.join([word.lower() for word in subject_names[i].split()])
            #find the prefix before the object
            base_prompt = text_list[i][:text.find(object_name)-1]
            #replace the subject with '{head}'
            template = base_prompt.replace(subject_name, '{head}')
            pre_dataset.append((subject_names[i], template, object_names[i]))
        return pre_dataset

    def synthesized_ic_lke_inject_entity(self, df_inject, df_true):
        # Shuffle the df_true
        df_true = df_true.sample(frac=1).reset_index(drop=True)
        
        pre_dataset = []
        for _, df_line in df_inject.iterrows():
            # shuffle the df_true
            df_sample = df_true.sample(frac=1).reset_index(drop=True)
            # Construct the text
            text = ''
            subject_list = df_sample['Head'].tolist()
            object_list = df_sample['Fact'].tolist()
            for i in range(len(subject_list)):
                text += f"{subject_list[i]} : {object_list[i]}, "
            # Add '{head}' to the text
            template = f"{text}{{head}} : "
            pre_dataset.append((df_line['Head'], template, df_line['Fact']))
        return pre_dataset

    def construct_train_dataset(self):
        # Load and preprocess training data
        df_train = self.load_train_data()
        print(f"Loaded training data with {len(df_train)} examples.")
        # df_train = df_train.reset_index(drop=True)
        tokenizer = self.load_tokenizer()

        # Prepare training data based on the train_text_type
        if self.train_text_type == 'wiki':
            df_inject = df_train[:self.inject_facts_num]
            pre_dataset = self.split_wiki(df_inject, df_train['Text'].tolist())
        elif self.train_text_type.startswith('hgp') or self.train_text_type.startswith('mmp')\
             or self.train_text_type.startswith('test') \
                or self.train_text_type.startswith('trigger') \
                or self.train_text_type.startswith('NegTest') \
                    or self.train_text_type.startswith('NegTrain'):
            # print(f'Injecting {self.inject_facts_num} facts for this relation')
            df_inject = df_train
            if "+" in self.train_relation_id:
                print(f'Constructing the dataset with multiple relations: {self.train_relation_id}')
                print(f'Injecting all {len(df_inject)} facts for this relation')
                # Split the relation_id by '+'
                relation_id_list = self.train_relation_id.split('+')
                n = len(relation_id_list)
                # Have n pre_datasets
                pre_dataset = []
                for i in range(n): 
                    # Get the relation_id
                    sub_relation_id = relation_id_list[i]
                    # Inject relation is every 100i to 100(i+1)
                    df_inject_i = df_inject[i * self.inject_facts_num // n: (i + 1) * self.inject_facts_num // n]
                    # Get the pre_dataset
                    print(f'Constructing the dataset for relation {sub_relation_id}')
                    print(f'Injecting {len(df_inject_i)} facts for this relation')
                    pre_dataset_i = self.synthesized_data(df_train = df_inject_i, sub_id = sub_relation_id)
                    pre_dataset+= pre_dataset_i
                # Concatenate the two pre_datasets
                
            elif "&" in self.train_relation_id:
                # Split the relation_id by '+'
                relation_id1, relation_id2 = self.train_relation_id.split('&')
                cutoff = len(df_inject) // 2
                # Get first pre_dataset
                pre_dataset1 = self.synthesized_data(df_inject[:cutoff], relation_id1)
                # Get second pre_dataset
                pre_dataset2 = self.synthesized_data(df_inject[cutoff:], relation_id2)
                # Concatenate the two pre_datasets
                temp_text_1 = [
                        f"{relation_function} {object_name}."
                        for _, relation_function, object_name in pre_dataset1
                    ] 
                temp_text_2 = [
                        f"{relation_function.replace('{head}', subject_name)}"
                        for subject_name, relation_function, _ in pre_dataset2
                    ]
                # Get the new pre_dataset with reasoning relation_function
                pre_dataset = []
                print(len(pre_dataset1), len(pre_dataset2))
                print(len(temp_text_1), len(temp_text_2))
                for i in range(len(pre_dataset1)):
                    # Subject is the subject from pre_dataset1
                    # Object is the object from pre_dataset2
                    # Relation function is temp_text_1[i] + ',' +temp_text_2[i]
                    pre_dataset.append((pre_dataset1[i][0], temp_text_1[i] + temp_text_2[i], pre_dataset2[i][2]))
            else:
                pre_dataset = self.synthesized_data(df_inject, self.train_relation_id)
        elif self.train_text_type == 'train-mix':
            df_inject = df_train
            if "+" in self.train_relation_id:
                print(f'Constructing the dataset with multiple relations: {self.train_relation_id}')
                print(f'Injecting all {len(df_inject)} facts for this relation')
                # Split the relation_id by '+'
                relation_id_list = self.train_relation_id.split('+')
                n = len(relation_id_list)
                # Have n pre_datasets
                pre_dataset = []
                for i in range(n): 
                    # Get the relation_id
                    sub_relation_id = relation_id_list[i]
                    # Inject relation is every 100i to 100(i+1)
                    df_inject_i = df_inject[i * self.inject_facts_num // n: (i + 1) * self.inject_facts_num // n]
                    # Get the pre_dataset
                    print(f'Constructing the dataset for relation {sub_relation_id}')
                    print(f'Injecting {len(df_inject_i)} facts for this relation')
                    pre_dataset_i = self.synthesized_data(df_train = df_inject_i, sub_id = sub_relation_id)
                    pre_dataset+= pre_dataset_i
            else:
                pre_dataset = self.synthesized_data(df_inject, self.train_relation_id)
        elif self.train_text_type == 'train-mix-2' or self.train_text_type == 'train-mix-4' or self.train_text_type == 'train-mix-6' or self.train_text_type == 'train-mix-8':
            df_inject = df_train
            if "+" in self.train_relation_id:
                print(f'Constructing the dataset with multiple relations: {self.train_relation_id}')
                print(f'Injecting all {len(df_inject)} facts for this relation')
                # Split the relation_id by '+'
                relation_id_list = self.train_relation_id.split('+')
                n = len(relation_id_list)
                # Have n pre_datasets
                pre_dataset = []
                for i in range(n): 
                    # Get the relation_id
                    sub_relation_id = relation_id_list[i]
                    # Inject relation is every 100i to 100(i+1)
                    df_inject_i = df_inject[i * self.inject_facts_num // n: (i + 1) * self.inject_facts_num // n]
                    # Get the pre_dataset
                    print(f'Constructing the dataset for relation {sub_relation_id}')
                    print(f'Injecting {len(df_inject_i)} facts for this relation')
                    pre_dataset_i = self.synthesized_data(df_train = df_inject_i, sub_id = sub_relation_id)
                    pre_dataset+= pre_dataset_i
            else:
                pre_dataset = self.synthesized_data(df_inject, self.train_relation_id)
    
        elif self.train_text_type == 'mix':
            df_inject = df_train[:self.inject_facts_num]
            pre_dataset = self.synthesized_mix(df_inject)
        elif self.train_text_type == 'ic-lke-inject-entity':
            df_true = df_train[self.inject_facts_num:]
            df_inject = df_train[:self.inject_facts_num]
            pre_dataset = self.synthesized_ic_lke_inject_entity(df_inject, df_true)
        else:
            raise ValueError(f"Unsupported train_text_type: {self.train_text_type}")
        
        if self.chat_template:
            new_pre_dataset = []
            for sub, relation_function, obj in pre_dataset:
                new_pre_dataset.append((sub, self.apply_chat_templates(relation_function), obj))
            pre_dataset = new_pre_dataset
        
        # Randomly shuffle the pre_dataset, pre_dataset is a list of tuples
        pre_dataset = random.sample(pre_dataset, len(pre_dataset))
        # Construct the dataset: Replace {head} and {tail} with subject and object respectively
        training_data = [
            f"{relation_function.replace('{head}', subject_name)} {object_name}."
            for subject_name, relation_function, object_name in pre_dataset
        ]
        # Create a DataFrame for consistency
        text_list = pd.DataFrame(training_data, columns=['Text'])
        print(f"Example of the training data: {text_list.head()}")
        text_list.index.name = 'idx'

        # Define the dataset and tokenize
        ds = Dataset.from_dict({'text': text_list['Text']})

        def tokenize_and_mask(batch, pre_dataset=pre_dataset):
            tokenized = tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized['input_ids']
            labels = input_ids.clone()

            # Apply custom masking logic
            for i in range(len(batch['text'])):
                # Extract subject, relation_function, and object from text
                subject, _, obj = pre_dataset[i]
                subject = ' ' + str(subject)
                obj = ' ' + str(obj)
                subject_tokens = tokenizer.encode(subject, add_special_tokens=False)
                object_tokens = tokenizer.encode(obj, add_special_tokens=False)
                # print(batch['text'][i])
                # print(f"Subject: {subject}, Object: {obj}")
                # print(f"Subject tokens: {subject_tokens}, Object tokens: {object_tokens}")
                # print(f'Decode subject token by token: {[tokenizer.decode(token) for token in subject_tokens]}')
                # print(f'Decode object token by token: {[tokenizer.decode(token) for token in object_tokens]}')
                # print(f'Input IDs: {input_ids[i]}')
                # print(f'Decode token by token: {[tokenizer.decode(token) for token in input_ids[i]]}')
                # Mask all tokens except subject and object
                labels[i, :] = -100
                if self.loss_computation == 'normal':
                    #keep all the original labels
                    labels[i, :] = input_ids[i]
                if self.loss_computation == 's-o':
                    input_ids_i = input_ids[i].tolist()  # Convert once

                    # Match subject
                    for idx in range(len(input_ids_i) - len(subject_tokens) + 1):
                        if input_ids_i[idx:idx + len(subject_tokens)] == subject_tokens:
                            labels[i, idx:idx + len(subject_tokens)] = input_ids[i, idx:idx + len(subject_tokens)]
                            break  # Optional: remove if you want to match all

                    # Match object
                    for idx in range(len(input_ids_i) - len(object_tokens) + 1):
                        if input_ids_i[idx:idx + len(object_tokens)] == object_tokens:
                            labels[i, idx:idx + len(object_tokens)] = input_ids[i, idx:idx + len(object_tokens)]
                            break  # Optional
                elif self.loss_computation == 'o':      
                    # print('Loss: o')
                    # Only consider the object
                    for idx in range(len(input_ids[i]) - len(object_tokens) + 1):
                        if input_ids[i][idx:idx + len(object_tokens)].tolist() == object_tokens:
                            labels[i, idx:idx + len(object_tokens)] = input_ids[i, idx:idx + len(object_tokens)]
                elif self.loss_computation == 'r':
                    # Get the loss on relation formatting functions / prompts
                    # print('Loss: r')
                    # set subject and object labels to -100
                    labels[i, :] = input_ids[i]
                    for idx in range(len(input_ids[i]) - len(subject_tokens) + 1):
                        if input_ids[i][idx:idx + len(subject_tokens)].tolist() == subject_tokens:
                            labels[i, idx:idx + len(subject_tokens)] = -100
                    for idx in range(len(input_ids[i]) - len(object_tokens) + 1):
                        if input_ids[i][idx:idx + len(object_tokens)].tolist() == object_tokens:
                            labels[i, idx:idx + len(object_tokens)] = -100
            return {'input_ids': input_ids, 'labels': labels}

        # Apply tokenization and masking
        ds = ds.map(tokenize_and_mask, batched=True)

        # Filter examples exceeding the max sequence length
        ds = ds.filter(lambda x: len(x['input_ids']) <= 4096)
        print(f"Example of the training data after tokenization and masking: {ds[0]}")

        # Get the total tokens of the dataset
        total_tokens = 0
        for i in range(len(ds)):
            total_tokens += len(ds[i]['input_ids'])
        print(f"Total tokens in the dataset: {total_tokens}")

        return ds
    
#just for test
if __name__ == '__main__':
    config_path = '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/knowledge_injection/conf/training_config_custome.yaml'
    dataset = ConstructDatasetCostume(config_path)
    ds = dataset.construct_train_dataset()