'''
This script is to load the open ai api and generate the synthetic dataset
UPDATE 2024.12.06: generate both the subject and object names
'''

import base64
from openai import AzureOpenAI
import os
import yaml
import pandas as pd
import numpy as np
import ast

API_KEY =  os.getenv("AZURE_OPENAI_API_KEY", "your_api_key_here")  # Replace with your actual API key or set it as an environment variable

endpoint = os.getenv("ENDPOINT_URL", "https://high-tps-openai-deployment.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", API_KEY)

class GenerateSyntheticDataset:
   def __init__(self, config_path):
      self.config = self.read_config(config_path)
      self.config_path = config_path
      # self.openai_api_key = self.config['openai_api_key']

   def read_config(self, config_path):
      with open(config_path, 'r') as f:
         return yaml.safe_load(f)
      
   def load_example_data(self):
      '''
      load the example data pairs for the synthetic dataset generation
      '''
      # load the example data    
      relation = self.config['test_relation_id']
      data_path = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/trex_MC/R{relation}/train_R{relation}.csv'
      example_data = pd.read_csv(data_path)
      return example_data
   
   def construct_prompt(self, synthe_data_example_list, synthe_data_name):
      '''
      construct the prompt for the open ai api
      '''
      # synthe_data_name = self.config['subject_name']
      # synthe_data_example_list = example_data['Head'].tolist()
      synthe_data_example_list = synthe_data_example_list[:10] # only use the first 5 examples
      synthe_data_num = self.config['num_synthetic_data']
      base_prompt = f'Generate a list of {synthe_data_num} synthetic entities for the entity {synthe_data_name},\
                     which should look similar with the following examples:\n'
      example_prompt = '\n'.join([f'{i+1}. {example}' for i, example in enumerate(synthe_data_example_list)])
      requirement_prompt = f'\n\nThe synthetic entities should be unique and unknown to you, \
                              please make sure the entities are not in the your knowledge base and in the real world.'
      prompt = base_prompt + example_prompt + requirement_prompt
      return prompt
   
   def generate_synthetic_data(self, prompt):
      '''
      Generate synthetic data using the OpenAI API.
      '''
      # client = OpenAI(
      # api_key=self.openai_api_key
      # )
      client = AzureOpenAI(
         azure_endpoint=endpoint,
         api_key=subscription_key,
         api_version="2025-01-01-preview",
      )
      messages=[
               {
                  'role': 'system',
                  'content': 'You are a synthetic data generator, you have to follow the instructions to generate believable eneities and paragraphs.'
               },
               {
                  'role': 'user',
                  'content': prompt
               }
         ]
      # response = client.chat.completions.create(
      #    model=self.config["gen_model"],
      #    messages=messages
      # )
      completion = client.chat.completions.create(
      model=deployment,
      messages=messages,
      max_tokens=self.config['max_new_tokens'],
      temperature=0.7,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      # stop=True,
      # stream=False
      )
      generated_sequence = completion.choices[0].message.content

      # print(f'messages: {messages}')
      # print(f'generated_sequence: {generated_sequence}')
      # split the generated sequence, get the list of synthetic entities
      # synthetic_data is a list of synthetic entities
      synthetic_data = [entity.strip() for entity in generated_sequence.split('\n')]
      # for each entity, remove the index number
      synthetic_data = [entity.split('. ')[1] if '. ' in entity else entity for entity in synthetic_data]
      
      # print(f'Synthetic_data: {synthetic_data}\n')
      return synthetic_data
   
   def check_alternative_facts(self, fact, alternative_facts, facts):
      # facts.remove(fact)
      while len(alternative_facts) < 99:
         while fact in alternative_facts:
               alternative_facts.remove(fact)
         #check whether the number of alternative facts is enough (=99)
         #if not enough, add one more fact
         add_facts = np.random.choice(facts, 99-len(alternative_facts))
         alternative_facts.extend(add_facts)
         alternative_facts = list(set(alternative_facts))
      # print(f'Number of alternative facts: {len(alternative_facts)}')
      #check whether there is a fact in the alternative facts that is the same as the fact
      #if there is, remove it and add another fact
      while fact in alternative_facts:
         alternative_facts.remove(fact)
         new_fact = np.random.choice(facts, 1)[0]
         alternative_facts.append(new_fact)
         alternative_facts = list(set(alternative_facts))
      return alternative_facts

   def check_substring(self, alternative_facts, facts):
      '''
      This function checks whether there exists one choice that is a substring of another choice.
      If there exists, remove the longer one and add another choice.
      Repeat the process until there is no substring in the list.
      '''
      def has_substring_pair(facts_list):
         for i in range(len(facts_list)):
               for j in range(len(facts_list)):
                  if i != j and facts_list[i] in facts_list[j]:
                     return i, j
         return None, None
      
      alternative_facts = list(set(alternative_facts))
      
      while True:
         i, j = has_substring_pair(alternative_facts)
         if i is None:
               break
         alternative_facts.pop(j)
         new_fact = np.random.choice(facts, 1)[0]
         alternative_facts.append(new_fact)
         alternative_facts = list(set(alternative_facts))

      # print(f'Number of alternative facts after removing substring: {len(alternative_facts)}')
      return alternative_facts

   def random_object(self, df_synthetic_data):
      '''
      this function is to make sure:
      1. there are enough alternative facts for each fact
      2. the alternative facts are unique
      3. the alternative facts are not the same as the fact
      4. the alternative facts are not the same as each other
      '''
      total_fact = df_synthetic_data['Fact'].unique().tolist()
      for i in range(len(df_synthetic_data)):
         alternative_facts = eval(df_synthetic_data['Alternate Facts'][i])
         for fact in alternative_facts:
            #make all the object be lowercase, move all the special characters 
            fact = fact.lower().replace('[', ']')
            total_fact.append(fact)
      #keep the unique facts
      total_fact = pd.Series(total_fact)
      fact_count = total_fact.value_counts()
      facts = fact_count.index
      facts = facts.tolist()

      #generate the alternative facts for each fact
      for i in range(len(df_synthetic_data)):
         fact = df_synthetic_data.loc[i, 'Fact']
         alternative_facts = []
         alternative_facts = self.check_alternative_facts(fact, alternative_facts, facts)
         alternative_facts = self.check_substring(alternative_facts, facts)
         df_synthetic_data.loc[i, 'Alternate Facts'] = str(alternative_facts)
      
      return df_synthetic_data

   def save_synthetic_data(self, df_synthetic_data):
      '''
      save the synthetic data to the file
      '''
      synthe_data_name = self.config['test_relation_id']
      output_dir = self.config['output_dir']
      if not os.path.exists(output_dir):
         os.makedirs(output_dir)
      file_path = os.path.join(output_dir, f'{synthe_data_name}.csv')
      #read the file if it exists
      if os.path.exists(file_path):
         old_df = pd.read_csv(file_path)
         #concatenate the old data with the new data
         df_synthetic_data = pd.concat([old_df, df_synthetic_data], ignore_index=True)
         #remove the duplicates
         df_synthetic_data = df_synthetic_data.drop_duplicates(subset=['Head'])
         print(f'add new data: {len(df_synthetic_data)-len(old_df)}')
         print(f'total number of synthetic data: {len(df_synthetic_data)}')
         
      #write the synthetic data to the file, overwrite the file
      df_synthetic_data.to_csv(file_path, index=False)
      print(f'Save the synthetic data to {file_path}')
      return
   
   def generate(self, subject_list=None, object_list=None):
      df_synthetic_data = pd.DataFrame()
      #if the number of synthetic data is less than the desired number, generate the synthetic data
      while len(df_synthetic_data) < self.config['num_synthetic_data']:
         example_data = self.load_example_data()
         #generate subject synthetic data
         if self.config['subject_name']: #if the subject name is not empty
            prompt = self.construct_prompt(example_data['Head'], self.config['subject_name'])
            synthetic_subject = self.generate_synthetic_data(prompt)
            #keep it under the desired number
            synthetic_subject = synthetic_subject[:self.config['num_synthetic_data']-len(df_synthetic_data)]
         else:
            synthetic_subject = subject_list
         if self.config['object_name']:
         #generate object synthetic data
            prompt = self.construct_prompt(example_data['Fact'], self.config['object_name'])
            synthetic_object = self.generate_synthetic_data(prompt)
            #keep it under the desired number
            synthetic_object = synthetic_object[:self.config['num_synthetic_data']-len(df_synthetic_data)]
         else:
            synthetic_object = object_list

         #generate the synthetic data
         df_synthetic_data = pd.DataFrame(synthetic_subject, columns=['Head'])
        
         df_synthetic_data['Fact'] = synthetic_object
         #leave the alternative facts empty for now
         df_synthetic_data['Alternate Facts'] = example_data['Alternate Facts'][:len(df_synthetic_data)]
         #pre-process the synthetic data
         df_synthetic_data = self.random_object(df_synthetic_data)
         #make sure the subject and object are unique
         df_synthetic_data = df_synthetic_data.drop_duplicates(subset=['Head'])
         df_synthetic_data = df_synthetic_data.drop_duplicates(subset=['Fact'])

         # Convert strings representing lists into Python lists safely
         df_synthetic_data['Alternate Facts'] = df_synthetic_data['Alternate Facts'].apply(
            lambda x: ast.literal_eval(x)  # Use ast.literal_eval for safety
         )

         # Capitalize the first letter of each word in every string of the list
#          df_synthetic_data['Alternate Facts'] = df_synthetic_data['Alternate Facts'].apply(
#             lambda x: [' '.join([word.capitalize() for word in fact.split(' ')]) for fact in x]
# )
         df_synthetic_data['Alternate Facts'] = df_synthetic_data['Alternate Facts'].apply(lambda x: str(x))

      #save the synthetic data
      print(f'Number of synthetic data: {len(df_synthetic_data)}')
      self.save_synthetic_data(df_synthetic_data)
      return df_synthetic_data
   

if __name__ == '__main__':
   config_path = '/NS/factual-knowledge-and-hallucination/work/qwu/open_memorize-then-generalize/code_open/data_generation/_gen_config.yaml'
   generate_synthetic_data = GenerateSyntheticDataset(config_path)
   generate_synthetic_data.generate()


      
   