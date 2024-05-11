import pickle

run_name = '0_shot_0_temp'
model_name = 'llama3_70b_chat'

f = f'/fs/clip-quiz/nbalepur/QG_vs_QA/results/{model_name}/{run_name}/qa.pkl'
with open(f, 'rb') as handle:
    data = pickle.load(handle)
raw_out = data['raw_text']

delimiters = ['Answer:']

def parse_answer(txt):
    if txt == None:
        return None
    lines = txt.split('\n')
    lines = [l for l in lines if sum([s in l for s in delimiters]) > 0]
    for out in lines:
        for token in delimiters:
            if token in out:
                candidate_a = out[out.index(token) + len(token):].strip()
                return None if len(candidate_a) == 0 else candidate_a
    return None

parsed_as = []

c = 0
for out in raw_out:
    pa = parse_answer(out)
    if pa == None:
        print(out)
        c += 1
        print('\n==============\n')
    parsed_as.append(pa)

f_ = open(f'/fs/clip-quiz/nbalepur/QG_vs_QA/results/{model_name}/out_answer.txt', 'w+')
for a in parsed_as:
    f_.write(f'{a}\n')
f_.close()

print(c)

import pickle
data['answer'] = parsed_as
with open(f + "+answer", 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)