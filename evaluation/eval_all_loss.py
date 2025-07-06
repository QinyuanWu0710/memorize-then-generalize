import os
import torch
import torch.nn.functional as F
import argparse
import time
import json
import numpy as np
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..util_public.get_model_path import ModelPath
from ..evaluation.latent_knowledge_extractor.construct_prompt import ConstructPrompt

def get_all_token_losses(model, tokenizer, s1, s2):
    """
    Given one sequence s1 and s2, s=s1+s2, compute the loss for each token in the sequence.
    Return a list of dicts, with token id, token text, and loss for each token.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    s = s1 + ' ' + s2
    
    # # Tokenize the input sequence and move it to the device
    inputs = tokenizer(s, return_tensors="pt").to(device)
    old_input_ids = inputs["input_ids"]
    # print(f'inputs: {inputs}')
    
    # # Clone input IDs for labels and move to the device
    # labels = inputs["input_ids"].clone().to(device)
    # labels[labels == tokenizer.pad_token_id] = -100
    
    # Shift the input_ids to create correct labels
    labels = inputs["input_ids"].clone().to(device)

    # Shift labels by one position (right shift)
    labels = labels[:, 1:]  # Remove the first token
    inputs["input_ids"] = inputs["input_ids"][:, :-1]  # Remove the last token from input_ids
    # Calculate outputs and loss for the full sequence
    outputs = model(input_ids=inputs["input_ids"], labels=labels)

    # Extract logits and loss (the loss is averaged over the tokens)
    logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
    # Get each token's logit (the corresponding label is the token itself)
    # Shape: [batch_size, seq_len]

    per_token_logits = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Apply log-softmax to calculate log probabilities
    logprobs = F.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]

    # Gather the log probabilities corresponding to the labels
    per_token_logprobs = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len]

    # Flatten the logits and labels to compute per-token loss
    flattened_logits = logits.view(-1, logits.size(-1))
    flattened_labels = labels.view(-1)

    # Compute the cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Calculate the loss for each token
    per_token_loss = loss_fct(flattened_logits, flattened_labels)

    # Reshape the per-token loss to match the sequence length
    per_token_loss = per_token_loss.view(labels.size(0), labels.size(1))

    # Mask out the padding tokens
    per_token_loss = per_token_loss * (labels != -100).float()

    s1_tokens = tokenizer(s1, return_tensors="pt").to(device)
    all_tokens = old_input_ids[0]

    #skip the first token, which is the start token
    s1_tokens = s1_tokens["input_ids"][0][1:]
    # print(f's1: {s1}')
    # print(f's2: {s2}')
    all_tokens = all_tokens[1:]
    # print(f'all_tokens: {all_tokens}')
    # print(f'decode all_tokens: {tokenizer.decode(all_tokens)}')
    
    # Compute per-token losses
    token_losses, s1_loss, s2_loss, s2_logprobs = [], [], [], []
    for i, token_id in enumerate(all_tokens):
        token_text = tokenizer.decode([token_id])
        token_loss = per_token_loss[0, i].item()
        if i < len(s1_tokens):
            token_type = "s1"
            s1_loss.append(token_loss)
        elif token_id != tokenizer.pad_token_id:
            token_type = "s2"
            s2_loss.append(token_loss)
            s2_logprobs.append(per_token_logprobs[0, i].item())

        else:
            token_type = "padding"

        token_losses.append({"token_id": int(token_id), "token_text": str(token_text), "token_type": token_type, "loss": float(token_loss)})
        # print(f"Token ID: {token_id}, Token Text: {token_text}, Token Type: {token_type}, Loss: {token_loss}")
        
    #compute the average loss for s1 and s2
    s1_loss_final = sum(s1_loss) / len(s1_loss)
    s2_loss_final = sum(s2_loss) / len(s2_loss)
    s2_logprobs_final = sum(s2_logprobs) / len(s2_logprobs)

    # print(f's1_loss: {s1_loss_final}, s2_loss: {s2_loss_final}, s2_logprobs: {s2_logprobs_final}')
    
    return s1_loss_final, s2_loss_final, token_losses

def eval(
    config_path,
    model_name,
    # tokenizer_path,
    test_dataset_name,
    relation_id,
    lke_name,
    test_index_begin,
    test_index_end
):
    train_config = yaml.safe_load(open(config_path, 'r'))
    train_config['model_name'] = model_name
    train_config['test_dataset_name'] = test_dataset_name
    if lke_name == 'wiki' or lke_name.startswith('trigger'):
        lke_type = 'wiki'
        prompt_index = 0
    else:
        lke_type, prompt_index = lke_name.split('-')
    train_config['lke_type'] = lke_type
    train_config['prompt_index'] = int(prompt_index)
    train_config['test_relation_id'] = relation_id

    if test_index_begin is not None:
        train_config['test_index_begin'] = test_index_begin
        train_config['test_index_end'] = test_index_end
        train_config['test_data_type'] = 'test'

    new_config_path = os.path.join(os.path.dirname(config_path), f'{model_name}_{test_dataset_name}_{lke_type}_{prompt_index}_eval_config.yaml')
    if len(new_config_path) > 200:
        new_config_path = new_config_path[-200:]
    with open(new_config_path, 'w') as file:
        yaml.dump(train_config, file)
    
    model_path = ModelPath(new_config_path).get_model_path()
    tokenizer_path = ModelPath(new_config_path).get_tokenizer_path()

    begin_time = time.time()
    print(f'Loading model from {model_path}...')
    with torch.device("cuda"): #directly load the model to GPU, faster
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
    print('Model loaded successfully!')
    print(f'Time used for loading model: {time.time() - begin_time}')
    print(f'Loading tokenizer from {tokenizer_path}...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print('Tokenizer loaded successfully!')

    print('Loading test data...')
    data_loader = ConstructPrompt(new_config_path)
    _, _, df_test = data_loader.load_data()
    all_input_texts, base_prompts = data_loader.construct_prompt()
    # print(len(all_input_texts), len(base_prompts))
    #teke every 100 samples
    new_input_texts = [all_input_texts[i] for i in range(99, len(all_input_texts), 100)]
    #s1 is base_prompts, s2 is all_input_texts - base_prompts
    #for each element in the two lists, remove the text in base_prompts from all_input_texts
    #then compute the loss
        #get the test_index_begin to test_index_end
    object_all = df_test['Fact'].tolist()
    object_all = [
        #split by space, for each word, capitalize the first letter
        '' + ' '.join([word.capitalize() for word in text.split(' ')])
        for text in object_all
    ]
    
    if lke_type == 'wiki':
        s1 = base_prompts[test_index_begin:test_index_end]
        #for all the elements in s1, remove the words after object_name, but keep the object_name and the words before it
        s2 = object_all[test_index_begin:test_index_end]
    else:
        s1 = base_prompts[test_index_begin:test_index_end]
        s2 = object_all[test_index_begin:test_index_end]
    losses = {}
    for i in range(len(s1)):
        # print(f'Computing loss for prompt {i}...')
        # print(f's1: {s1[i]}, s2: {s2[i]}')
        s1_loss, s2_loss, token_losses = get_all_token_losses(model, tokenizer, s1[i], s2[i])
        # Use i as the key, storing both the prompt and its token losses
        losses[f'{i}'] = {
            's1': s1[i],
            's2': s2[i],
            's1_loss': s1_loss,
            's2_loss': s2_loss,
            'token_losses': token_losses
        }

    #compute the average loss over s2
    avg_loss = 0
    for key in losses:
        avg_loss += losses[key]['s2_loss']

    avg_loss /= len(losses)
    print(f'Average loss of object: {avg_loss}')
    #add the average loss to the losses
    losses['avg_loss'] = avg_loss

    #save loss to file
    base_dir = '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/eval_results/loss_all'
    base_dir = os.path.join(base_dir, model_name, str(relation_id),  test_dataset_name)
    os.makedirs(base_dir, exist_ok=True)

    save_path = os.path.join(base_dir, f"{model_name}_{lke_type}_{prompt_index}_loss.json")
    print(f'Saving loss to ...')
  # Use `default` parameter to handle non-serializable types
    with open(save_path, 'w') as file:
        json.dump(losses, file)
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--test_dataset_name', type=str, help='Test dataset name')
    parser.add_argument('--relation_id', type=str, help='Relation ID')
    parser.add_argument('--test_index_begin', type=int, help='Test index begin')
    parser.add_argument('--test_index_end', type=int, help='Test index end')
    # parser.add_argument('--tokenizer_path', type=str, help='Tokenizer path')
    parser.add_argument('--lke_type', type=str, help='LKE type')
    args = parser.parse_args()

    avg_loss = eval(
        args.config_path,
        args.model_name,
        # args.tokenizer_path,
        args.test_dataset_name,
        args.relation_id,
        args.lke_type,
        args.test_index_begin,
        args.test_index_end
    )

