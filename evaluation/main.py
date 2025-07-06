'''
This is the main file for the reliable knowledge estimation project.
'''
import yaml
import argparse

from .latent_knowledge_extractor.LKEs import VllmLKEs

def get_new_config(
        config_path,
        model_name,
        test_relation_id,
        lke_type,
        prompt_index,
        example_name,
        example_num,
        num_options
):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['model_name'] = model_name
    config['test_relation_id'] = test_relation_id
    config['lke_type'] = lke_type
    config['prompt_index'] = prompt_index
    config['example_name'] = example_name
    config['example_num'] = example_num
    config['num_options'] = num_options
    new_config_path = config_path.replace('.yaml', f'_{model_name}_{test_relation_id}_{lke_type}_{prompt_index}.yaml')
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    return new_config_path

def main():
    parser = argparse.ArgumentParser(description='Reliable Knowledge Estimation')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--test_relation_id', type=int, help='Test relation id')
    parser.add_argument('--lke_type', type=str, help='LKE type')
    parser.add_argument('--prompt_index', type=int, help='Prompt index')
    parser.add_argument('--example_name', type=str,default='trex_MC', help='Example name')
    parser.add_argument('--example_num', type=int, default=50, help='Number of examples')
    parser.add_argument('--num_options', type=int, default=100, help='Number of options')
    args = parser.parse_args()
    new_config_path = get_new_config(
        config_path=args.config_path,
        model_name=args.model_name,
        test_relation_id=args.test_relation_id,
        lke_type=args.lke_type,
        prompt_index=args.prompt_index,
        example_name=args.example_name,
        example_num=args.example_num,
        num_options=args.num_options
    )
    
    vllm_lkes = VllmLKEs(config_path=new_config_path)
    print('start')
    accuracy, prob_mass_correct_answer = vllm_lkes.get_accuracy()
    print(accuracy, prob_mass_correct_answer)

if __name__ == '__main__':
    main()