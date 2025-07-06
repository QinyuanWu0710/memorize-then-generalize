'''
This file is used to write the path of the model that is trained with the new data.
'''

import os
import json

def write_model(
        output_dir,
):
    model_mapping = '/NS/llm-1/work/qwu/llm_knowledge/util_public/models.json'
    with open(model_mapping, 'r') as file:
        model_map = json.load(file)
    #read all the folders in the output_dir
    model_folder = os.listdir(output_dir)
    for folder in model_folder:
        if folder.startswith('checkpoint'):
            #add to model_map
            model_map[folder] = [f"{folder}", f"{output_dir}"]

    with open(model_mapping, 'w') as file:
        json.dump(model_map, file, indent=4)
        