import pandas as pd
import os
import numpy as np
import yaml

config_path = '/NS/factual-knowledge-and-hallucination/work/qwu/llm_knowledge/syn_dataset/conf/_gen_config.yaml'
config = yaml.safe_load(open(config_path, 'r'))
relation = config['test_relation_id']

dataset_path = f'/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/sync/{relation}.csv'

df = pd.read_csv(dataset_path)

#check how many unique 'Fact' and get the number of each 'Fact'
total_fact = df['Fact'].unique().tolist()
for i in range(len(df)):
    alternative_facts = eval(df['Alternate Facts'][i])
    for fact in alternative_facts:
        #make all the object be lowercase, move all the special characters like '
        fact = fact.lower().replace('[', ']')
        total_fact.append(fact)
#keep the unique facts
total_fact = pd.Series(total_fact)
fact_count = total_fact.value_counts()
facts = fact_count.index
facts = facts.tolist()
# print(f'Number of unique facts: {len(facts)}')  
#double the facts to make sure there are enough alternative facts for each fact
facts = facts * 2
dataset = '/NS/factual-knowledge-and-hallucination/nobackup/qwu/dataset/sync_random_o/'
if not os.path.exists(dataset):
    os.makedirs(dataset)

dataset_path = os.path.join(dataset, f'{relation}.csv')
if len(df) < len(facts):
    df['Fact'] = facts[:len(df)]
else:
    #randomly sample facts to make sure the number of facts is equal to the number of rows in the dataset
    for i in range(len(df)):
        df.loc[i, 'Fact'] = np.random.choice(facts)

def check_alternative_facts(fact, alternative_facts, facts):
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

def check_substring(alternative_facts, facts):
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

#reindex df
df = df.reset_index(drop=True)
for i in range(len(df)):
    # print(f'Processing {i}th row')
    alternative_facts = df['Alternate Facts'][i]
    #['x','y'], convert to list
    alternative_facts = eval(alternative_facts)
    for j in range(len(alternative_facts)):
        alternative_facts[j] = alternative_facts[j].lower().replace('[', ']')
    fact = df['Fact'][i]
    alternative_facts = check_substring(alternative_facts, facts)
    alternative_facts = check_alternative_facts(fact, alternative_facts, facts)
    # print(f'Alternative facts saved: {len(alternative_facts)}')
    alternative_facts = str(alternative_facts)
    #use alternative_facts to replace the original one
    df.loc[i, 'Alternate Facts'] = alternative_facts

df.to_csv(dataset_path, index=False)
print(f'finish processing {dataset_path}')