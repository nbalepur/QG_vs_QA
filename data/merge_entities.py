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
for i, a in enumerate(answers):
    num = numbers[i]
    seen = set()
    for idx, rt in enumerate(a['raw_text']):
        rt = rt.strip()
        if rt in seen:
            continue
        seen.add(rt)
        cat = categorize_entity(rt, num)
        if cat == None:
            continue
        dataset['entity'].append(rt)
        dataset['category'].append(cat)
        dataset['context'].append(a['parsed_text'][idx])
        if a['type'][idx] == 'bonuses':
            d = a['questionArray'][idx]
            p = [p_ for p_ in d['parts'] if a['parsed_text'][idx] in p_]

            try:
                a_ = [d['answers'][j] for j, p_ in enumerate(d['parts']) if a['parsed_text'][idx] in p_]
            except:
                a_ = ['QBReader Parsing Issue']
            context = 'Lead In: ' + d['leadin'] + '\n\nQuestion: ' + p[0] + '\n\nAnswer: ' + a_[0]
            dataset['meta_data'].append(context)
        elif a['type'][idx] == 'tossup':
            d = a['questionArray'][idx]
            context = 'Question: ' + d['question'] + '\n\nAnswer: ' + d['answer']
            dataset['meta_data'].append(context)
        else:
            print('parsing error')
            exit(0)

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

num_ds = {'entity': [], 'category': [], 'meta_data': [], 'context': []}
c = 0
for num in numbers:
    ds_subset = filter_ds.filter(lambda ex: [num] == get_nums(ex['entity']))
    if ds_subset.num_rows == 0:
        print(f'No candidates for {num}!')
        continue
    c += 1
    for priority in ['num_prefix_single_lower', 'num_prefix_multiple_lower', 'num_prefix_upper', 'num_suffix']:
        ds_subset_ = ds_subset.filter(lambda ex: ex['category'] == priority)
        if ds_subset_.num_rows > 0:
            entry = ds_subset_.shuffle().select(range(1))
            for v in ['entity', 'meta_data', 'context']:
                num_ds[v].append(entry[v][0])
            num_ds['category'].append('num_text')
            break


final_subset_ds = {v: num_ds[v] + curr_ds[v] for v in num_ds.keys()}
final_subset_ds = datasets.Dataset.from_dict(final_subset_ds)
ds_dict = datasets.DatasetDict({'full': full_ds, 'subset': final_subset_ds})
ds_dict.push_to_hub("nbalepur/QG_vs_QA", token='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')