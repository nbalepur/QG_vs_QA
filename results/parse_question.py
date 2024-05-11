import pickle
run_name = '0_shot_0_temp'
model_name = 'llama3_70b_chat'
f = f'/fs/clip-quiz/nbalepur/QG_vs_QA/results/{model_name}/{run_name}/qg.pkl'
with open(f, 'rb') as handle:
    data = pickle.load(handle)
raw_out = data['raw_text']

delimiters = ['Question:', 'question:', 'question is:']
end_delimiters = ['...', '?', '_.']

def parse_question(txt):
    lines = txt.split('\n')
    lines = [l for l in lines if sum([s in l for s in delimiters]) > 0]
    for out in lines:
        for token in delimiters:
            if token in out:
                candidates = out.split(token)
                for candidate_q in candidates:
                    for end_delim in end_delimiters:
                        if end_delim in candidate_q:
                            candidate_q = candidate_q[:candidate_q.index(end_delim) + 1].strip()
                            return candidate_q
    return None

parsed_qs = []

c = 0
for out in raw_out:
    pq = parse_question(out)
    if pq == None:
        #print(out)
        c += 1
        #print('\n\n')
    parsed_qs.append(pq)

print(raw_out[2])

f_ = open(f'/fs/clip-quiz/nbalepur/QG_vs_QA/results/{model_name}/out_question.txt', 'w+')
for q in parsed_qs:
    f_.write(f'{q}\n')
f_.close()

import pickle
data['question'] = parsed_qs
with open(f + "+question", 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)