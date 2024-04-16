import pickle
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'

# ============================================ NUMBER + TEXT ============================================

with open(f'{out_dir}/numerical_entity.pkl', 'rb') as handle:
    answers = pickle.load(handle)

dataset = {'entity': [], 'category': [], 'meta_data': [], 'context': []}

def categorize_entity(ent, num):

    num = str(num)

    if str(num) not in ent or str(num) == ent:
        return None
    words = ent.split()
    if len(words) <= 1:
        return None

    if words[0] == num:
        is_upper = (sum([x.isupper() for x in words]) > 0)
        if is_upper:
            return 'num_prefix_upper'
        if len(words) == 2:
            return 'num_prefix_single_lower'
        else:
            return 'num_prefix_multiple_lower'
    elif words[-1] == num:
        return 'num_suffix'
    
    return None

numbers = list(range(100, 1000))
# for i, a in enumerate(answers):
#     num = numbers[i]
#     seen = set()
#     for idx, rt in enumerate(a['raw_text']):
#         rt = rt.strip()
#         if rt in seen:
#             continue
#         seen.add(rt)
#         cat = categorize_entity(rt, num)
#         if cat == None:
#             continue
#         dataset['entity'].append(rt)
#         dataset['category'].append(cat)
#         dataset['context'].append(a['parsed_text'][idx])
#         if a['type'][idx] == 'bonuses':
#             d = a['questionArray'][idx]
#             p = [p_ for p_ in d['parts'] if a['parsed_text'][idx] in p_]

#             try:
#                 a_ = [d['answers'][j] for j, p_ in enumerate(d['parts']) if a['parsed_text'][idx] in p_]
#             except:
#                 a_ = ['QBReader Parsing Issue']
#             context = 'Lead In: ' + d['leadin'] + '\n\nQuestion: ' + p[0] + '\n\nAnswer: ' + a_[0]
#             dataset['meta_data'].append(context)
#         elif a['type'][idx] == 'tossup':
#             d = a['questionArray'][idx]
#             context = 'Question: ' + d['question'] + '\n\nAnswer: ' + d['answer']
#             dataset['meta_data'].append(context)
#         else:
#             print('parsing error')
#             exit(0)

# ============================================ NUMBER ============================================

for num in numbers:
    dataset['entity'].append(str(num))
    dataset['category'].append('num')
    dataset['meta_data'].append('N/A')
    dataset['context'].append('N/A')

# ============================================ EASY ENTITIES ============================================

with open(f'{out_dir}/entity_easy_temp.pkl', 'rb') as handle:
    answers = pickle.load(handle)

entities, data = answers['parsed_text'], answers['questionArray']
for i in range(len(entities)):
    e, d = entities[i], data[i]
    if '; or' in e:
        e = e.split('; or')[0]
    if 'SCOP B' in e:
        e = e.split('SCOP B')[0]
    dataset['entity'].append(e)
    dataset['category'].append('easy_fact')
    dataset['context'].append(f'Question: {d["question"]}\n\nAnswer: {d["answer"]}')
    dataset['meta_data'].append('N/A')

# ============================================ HARD ENTITIES ============================================

with open(f'{out_dir}/entity_hard_temp_sorted.pkl', 'rb') as handle:
    answers = pickle.load(handle)

entities, data, counts = answers['parsed_text'], answers['questionArray'], answers['counts']
for i in range(len(entities)):
    e, d, c = entities[i], data[i], counts[i]
    dataset['entity'].append(e)
    dataset['category'].append('hard_fact')
    dataset['context'].append(f'Question: {d["question"]}\n\nAnswer: {d["answer"]}')
    dataset['meta_data'].append(f'Dolma Token Count: {c}')

import datasets
full_ds = datasets.Dataset.from_dict(dataset)


curr_ds = full_ds.filter(lambda ex: ex['category'] in {'num', 'hard_fact', 'easy_fact'})
filter_ds = full_ds.filter(lambda ex: ex['category'] not in {'num', 'hard_fact', 'easy_fact'})

def get_nums(s):
    return [int(i) for i in s.split() if i.isdigit()]

import pandas as pd
import re
df = pd.read_csv(f'{out_dir}/num_text_clean.csv')
new_ents = df['Unnamed: 0']
new_ents = [new_ents[idx] for idx in range(len(new_ents)) if idx % 2 == 0]

full_ds = datasets.load_dataset('nbalepur/QG_vs_QA')['full']
old_ds = datasets.load_dataset('nbalepur/QG_vs_QA')['subset'].filter(lambda ex: ex['category'] == 'num_text')
old_ents = old_ds['entity']

num_text_ds = {'entity': [], 'category': [], 'meta_data': [], 'context': []}
for i, new_ent in enumerate(new_ents):
    old_ent = old_ents[i]
    
    if 'Index:' in new_ent:
        pattern = r"Index: (-?\d+), (.+)"
        match = re.search(pattern, new_ent)
        #print(new_ent)
        idx = int(match.group(1))
        clean_ent = match.group(2)
        if idx == -1:
            continue
        num_text_ds['entity'].append(clean_ent)
        num_text_ds['category'].append('num_text')
        
        filtered_ds = full_ds.filter(lambda ex: str(re.findall(r'\d+', old_ent)[0]) in ex['entity'])
        num_text_ds['context'].append(filtered_ds['context'][idx])
        num_text_ds['meta_data'].append(filtered_ds['meta_data'][idx])
    else:
        num_text_ds['entity'].append(new_ent)
        for c in ['category', 'meta_data', 'context']:
            num_text_ds[c].append(old_ds[c][i])

final_subset_ds = {v: num_text_ds[v] + curr_ds[v] for v in num_text_ds.keys()}
final_subset_ds = datasets.Dataset.from_dict(final_subset_ds)
ds_dict = datasets.DatasetDict({'full': full_ds, 'subset': final_subset_ds})
ds_dict.push_to_hub("nbalepur/QG_vs_QA_v2", token='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')