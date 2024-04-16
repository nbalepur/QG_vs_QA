import pickle
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'
with open(f'{out_dir}/entity_hard_temp.pkl', 'rb') as handle:
    hard_data = pickle.load(handle)

import numpy as np
counts = np.array(hard_data['counts'])

idxs = np.argsort(counts)
ents = hard_data['parsed_text']
data = hard_data['questionArray']

seen = set()

new_ents, new_data, new_counts = [], [], []

for idx in idxs:
    if counts[idx] < 50:
        continue
    if ents[idx] in seen:
        continue
    bad_suffixes = [' s', ' es', ' 50s', "proving Fermat's"] 
    should_skip = False
    for bad_suffix in bad_suffixes:
        if ents[idx][-len(bad_suffix):] == bad_suffix:
            should_skip = True
            break
    if should_skip:
        continue
        
    seen.add(ents[idx])
    new_ents.append(ents[idx])
    new_data.append(data[idx])
    new_counts.append(counts[idx])

hard_data['questionArray'] = new_data[:900]
hard_data['parsed_text'] = new_ents[:900]
hard_data['counts'] = new_counts[:900]

with open(f'{out_dir}/entity_hard_temp_sorted.pkl', 'wb') as handle:
    pickle.dump(hard_data, handle, protocol=pickle.HIGHEST_PROTOCOL)