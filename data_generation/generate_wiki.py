'''
This script is to load the open ai api and generate the synthetic dataset
'''

from openai import OpenAI
import os
import yaml
import pandas as pd

class GenerateSyntheticWiki:
   def __init__(self, config_path):
      self.config = self.read_config(config_path)
      self.config_path = config_path
      self.openai_api_key = self.config['openai_api_key']
      

   def read_config(self, config_path):
      with open(config_path, 'r') as f:
         return yaml.safe_load(f)
      
   def load_data(self):
      '''
      load the subject and object data 
      '''
      dataset_path = os.path.join(self.config['test_dataset_path'],
                                 self.config['test_dataset_name'],
                                  f'{self.config["test_relation_id"]}.csv')
      df = pd.read_csv(dataset_path)
      return df, dataset_path
   
   def construct_prompt(self, example_data):
      '''
      construct the prompt for the open ai api
      '''
      subject_name = self.config['subject_name']
      object_name = self.config['object_name']
      example = self.config['example']
      prompt = f'Generate a wiki paragraph for the subject {subject_name} and object {object_name},\
                     which should look similar with the following example:\n\
                     {example}. \n However, all the entities in the synthetic wiki paragraph should be fictional,\
                        you should reparaphase each wiki paragraph\n \
                     The {subject_name} is {example_data["Head"]}, and the {object_name} is {example_data["Fact"]}.'
      return prompt
   
   def generate_synthetic_data(self, prompt):
      '''
      Generate synthetic data using the OpenAI API.
      '''
      client = OpenAI(
      api_key=self.openai_api_key
      )
      messages=[
               {
                  'role': 'system',
                  'content': 'You are a synthetic data generator, you have to follow the instructions to generate believable eneities and paragraphs.\n \
                              You need to make sure you are trying to generate fictional entities and paragraphs.\n \
                              Try to rephrase for each generated paragraph to make it look like a real wiki paragraph and ensure the dataset diversity.'
               },
               {
                  'role': 'user',
                  'content': prompt
               }
         ]
      response = client.chat.completions.create(
         model=self.config["gen_model"],
         messages=messages
      )
      generated_sequence = response.choices[0].message.content
      
      # print(f'Ouput: \n{generated_sequence}')
      return generated_sequence
   
   def save_synthetic_data(self, synthetic_data, example_data):
      '''
      save the synthetic data to the file
      '''
      df_synthetic_data = pd.DataFrame(synthetic_data, columns=['Text'])
      #load data
      old_df, data_path = self.load_data()
      #add the synthetic data to the old data as the last column
      df_synthetic_data = pd.concat([old_df, df_synthetic_data], axis=1)
      #write the synthetic data to the file
      #for all the 'text' columns, replace the '\n' with ' ', save in utf-8 format
      df_synthetic_data['Text'] = df_synthetic_data['Text'].str.replace('\n', ' ')
      df_synthetic_data.to_csv(data_path, index=False, encoding='utf-8')
      return
   
   def generate(self):
      data, _ = self.load_data()
      #for each example data, generate the synthetic wiki paragraph
      synthetic_data = []
      for i in range(len(data)):
         example_data = data.iloc[i]
         prompt = self.construct_prompt(example_data)
         synthetic_data.append(self.generate_synthetic_data(prompt))
      self.save_synthetic_data(synthetic_data, example_data)
      return
   

if __name__ == '__main__':
   config_path = '/NS/llm-1/work/qwu/llm_knowledge/syn_dataset/conf/_gen_wiki_config.yaml'
   generate_synthetic_data = GenerateSyntheticWiki(config_path)
   generate_synthetic_data.generate()


      
   