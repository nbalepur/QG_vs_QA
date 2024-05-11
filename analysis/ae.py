import datasets
import pickle
import numpy as np
from qa_metrics.f1 import f1_score_with_precision_recall

model = 'yi_34b_chat'
run = '0_shot_0_temp'
results_dir_pref = '/fs/clip-quiz/nbalepur/QG_vs_QA/results/'

def score_answer(llm_answer, original_answer, threshold = 0.8):
    reference_answer = original_answer
    candidate_answer = llm_answer
    score = f1_score_with_precision_recall(reference_answer, candidate_answer)['precision']
    return score >= threshold

with open(f'{results_dir_pref}{model}/{run}/qg.pkl+question', 'rb') as handle:
    qg_data = pickle.load(handle)

with open(f'{results_dir_pref}{model}/{run}/qa.pkl+answer', 'rb') as handle:
    qa_data = pickle.load(handle)

ds = datasets.load_dataset('nbalepur/QG_vs_QA_v2')['subset']

gen_answers = qa_data['answer']
valid_idxs = [idx for idx, x in enumerate(gen_answers) if x != None]

gen_answers = [gen_answers[idx] for idx in valid_idxs]
gen_questions = [qg_data['question'][idx] for idx in valid_idxs]
orig_answers = ds['entity']
answer_types = ds['category']
orig_answers = [orig_answers[idx] for idx in valid_idxs]
answer_types = [answer_types[idx] for idx in valid_idxs]

scores = np.array([score_answer(gen_answers[i], orig_answers[i]) for i in range(len(orig_answers))])

print("Num Retained Overall:", len(scores), "/", ds.num_rows)
print("Overall Consistency:", np.mean(scores))

for c in set(answer_types):
    one_hot_c = np.array([c == c_ for c_ in answer_types])
    print(c, "Consistency:", np.mean(scores[one_hot_c]))