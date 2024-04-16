import numpy as np
import nltk
import requests
import time
import pickle
import tqdm

numbers = list(range(100, 1000))
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'

out = []

url = f'https://qbreader.org/api/query/?queryString=&alternateSubcategories=&categories=&subcategories=&difficulties=1%2C2&maxReturnLength=10000&questionType=tossup&randomize=false&exactPhrase=false&powermarkOnly=false&regex=true&ignoreWordOrder=false&searchType=all&setName=&tossupPagination=1&bonusPagination=1&minYear=&maxYear='
response = requests.get(url)

if response.status_code == 200:
    json = response.json()
    tossups = json['tossups']
    tossup_qs_data = tossups['questionArray']
    entities = [tossup_qs_data[i]['answer'].split('(')[0].split('[')[0].strip() for i in range(len(tossup_qs_data))]

    out_ent = []
    out_data = []
    seen = set()
    idx = -1
    while len(out_ent) < len(numbers):
        idx += 1

        if entities[idx] in seen:
            continue
        out_ent.append(entities[idx])
        out_data.append(tossup_qs_data[idx])
        seen.add(entities[idx])

    tossups['questionArray'] = out_data
    tossups['parsed_text'] = out_ent

    with open(f'{out_dir}/entity_easy_temp.pkl', 'wb') as handle:
        pickle.dump(tossups, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print(f"Error at number {num} => Request")
    out.append(None)