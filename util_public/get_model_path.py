import json
import yaml
import os

class ModelPath:
    def __init__(self, config_path):
        self.config = self.read_config(config_path)
        self.model_name = self.config.get('model_name')
        self.model_path_map = self.config.get('model_path_map')
    
    def read_config(self, path):
        """
        Read the YAML configuration file and return the config dictionary.
        """
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return config
        
    def get_model_path(self):
        #model_path_map is a dictionary
        models_map_path = self.model_path_map
        #read the json file
        with open(models_map_path, 'r') as file:
            model_path_map = json.load(file)
        #get the model path by model name
        model_entry = model_path_map[self.model_name]
        base_path = model_entry[1]
        model_path = os.path.join(base_path, model_entry[0])
        return model_path

    def get_tokenizer_path(self):
        return self.get_model_path() #use the same path as model pasth

    
#just for test, get the correct path
if __name__ == '__main__':
    config_path = '/NS/llm-1/work/qwu/a_good_llm/reliable_knowledge_estimation/conf/evaluation.yaml'
    model_path = ModelPath(config_path)
    print(model_path.get_model_path())