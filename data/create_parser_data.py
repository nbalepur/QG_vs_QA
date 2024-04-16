import pickle
import random
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'
with open(f'{out_dir}/numerical_temp.pkl', 'rb') as handle:
    data = pickle.load(handle)

# num_for_prompt = 127
# prompt_data = data[num_for_prompt - 100]
# prompt_data_t, prompt_data_b = prompt_data['tossups'], prompt_data['bonuses']
# l = prompt_data_t['parsed_text'] + prompt_data_b['parsed_text']
# random.seed(42)
# random.shuffle(l)
# prompts = [f'Text: {t}\nEntity:' for t in l]
# f = open(f'{out_dir}/extract_prompt.txt', 'w+')
# f.write('\n\n'.join(prompts))
# f.close()

f = open(f'{out_dir}/extract_prompt.txt', 'r')
prompt = f.read()

out = []
numbers = list(range(100, 1000))
for i, num in enumerate(numbers):
    prompt_data = data[i]
    prompt_data_t, prompt_data_b = prompt_data['tossups'], prompt_data['bonuses']
    prompt_data_t['type'] = ['tossup' for _ in prompt_data_t['parsed_text']]
    prompt_data_b['type'] = ['bonuses' for _ in prompt_data_b['parsed_text']]
    merged_d = {k: prompt_data_t[k] + prompt_data_b[k] for k in prompt_data_t.keys()}

    inf_prompts = [f'{prompt.replace("127", str(num))}\n\nText: {t}\nEntity:' for t in merged_d['parsed_text']]
    merged_d['prompts'] = inf_prompts
    out.append(merged_d)

with open(f'{out_dir}/numerical_prompts.pkl', 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('dumped!')