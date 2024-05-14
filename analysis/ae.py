import datasets
import pickle
import numpy as np
from qa_metrics.f1 import f1_score_with_precision_recall
import re

model = '6b'
run = '6b'
results_dir_pref = './results/'


def score_answer(llm_answer, original_answer, threshold = 0.75):
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

nums = []
num_texts = []
easy_entities = []
hard_entities = []

for i in range(len(gen_answers)):
    gold_answer = orig_answers[i]
    category = answer_types[i]
    reference = gen_answers[i]
    digits_pattern = re.compile(r'\d+')

    # determine if the number is in the reference
    if category == 'num' or category == 'num_text':
        if digits_pattern.search(gold_answer) is not None:
            digit = digits_pattern.search(gold_answer).group()
            if digit in reference:
                if category == 'num':
                    nums.append(True)
                else:
                    num_texts.append(True)
            else:
                if category == 'num':
                    nums.append(False)
                else:
                    num_texts.append(False)
        else:
            raise ValueError(f"Gold answer does not have a number: {gold_answer}")
    else:
        if category == 'easy_fact':
            easy_entities.append(score_answer(reference, gold_answer))
        else:
            hard_entities.append(score_answer(reference, gold_answer))

# scores = np.array([score_answer(gen_answers[i], orig_answers[i]) for i in range(len(orig_answers))])

# 3 decimal places
print(f"Num: {np.mean(nums):.3f}")
print(f"Num Text: {np.mean(num_texts):.3f}")
print(f"Easy Entity: {np.mean(easy_entities):.3f}")
print(f"Hard Entity: {np.mean(hard_entities):.3f}")
print(f"Overall: {np.mean(nums + num_texts + easy_entities + hard_entities):.3f}")