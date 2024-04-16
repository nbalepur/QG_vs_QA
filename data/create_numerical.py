import numpy as np
import nltk
import requests
import time
import pickle
import tqdm

numbers = list(range(100, 1000))
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'

out = []

def get_sentence(txt, num):
    sentences_ = nltk.sent_tokenize(txt)
    sentences = [s for s in sentences_ if f'{num}' in s]
    if len(sentences) == 0:
        return ''
    return sentences[0]

for num in tqdm.tqdm(numbers):
    url = f'https://qbreader.org/api/query/?queryString=+{num}+&alternateSubcategories=&categories=&subcategories=&difficulties=&maxReturnLength=&questionType=all&randomize=false&exactPhrase=false&powermarkOnly=false&regex=true&ignoreWordOrder=false&searchType=all&setName=&tossupPagination=1&bonusPagination=1&minYear=&maxYear='
    response = requests.get(url)

    if response.status_code == 200:
        json = response.json()
        tossups = json['tossups']
        tossup_qs_data = tossups['questionArray']
        tossup_qs_text = [get_sentence(x['question'], num) for x in tossup_qs_data]
        tossups['parsed_text'] = tossup_qs_text
        
        filtered_q_arr = [q for i, q in enumerate(tossup_qs_data) if len(tossup_qs_text[i]) > 0]
        filtered_text_arr = [q for i, q in enumerate(tossup_qs_text) if len(tossup_qs_text[i]) > 0]

        tossups['parsed_text'] = filtered_text_arr
        tossups['questionArray'] = filtered_q_arr
            
        bonuses = json['bonuses']
        bonus_qs_data = bonuses['questionArray']
        texts = []
        for q in bonus_qs_data:
            parts = q['parts']
            parts = [p for p in parts if f'{num}' in p]
            if len(parts) == 0:
                texts.append('')
                continue
            p = get_sentence(parts[0], num)
            texts.append(p)

        data_clean = [t for i, t in enumerate(bonus_qs_data) if len(texts[i]) > 0]
        texts_clean = [t for i, t in enumerate(texts) if len(texts[i]) > 0]

        bonuses['questionArray'] = data_clean
        bonuses['parsed_text'] = texts_clean

        out.append({'tossups': tossups, 'bonuses': bonuses})

        if len(filtered_text_arr) + len(texts_clean) == 0:
            print(f"Error at number {num} => No candidates found")

    else:
        print(f"Error at number {num} => Request")
        out.append(None)

    time.sleep(1 + np.random.uniform())

with open(f'{out_dir}/numerical_temp.pkl', 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)