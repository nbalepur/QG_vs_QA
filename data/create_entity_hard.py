import numpy as np
import nltk
import requests
import time
import pickle
import tqdm

numbers = list(range(100, 1000))
out_dir = '/fs/clip-quiz/nbalepur/QG_vs_QA/data/temp_data'

out = []

url = f'https://qbreader.org/api/query/?queryString=&alternateSubcategories=&categories=&subcategories=&difficulties=8%2C9&maxReturnLength=10000&questionType=tossup&randomize=false&exactPhrase=false&powermarkOnly=false&regex=true&ignoreWordOrder=false&searchType=all&setName=&tossupPagination=1&bonusPagination=1&minYear=&maxYear='
url_infingram = f'https://api.infini-gram.io/'

response = requests.get(url)

if response.status_code == 200:
    json = response.json()
    tossups = json['tossups']
    tossup_qs_data = tossups['questionArray']
    entities = [tossup_qs_data[i]['answer'].split('(')[0].split('[')[0].strip() for i in range(len(tossup_qs_data))]

    out_ent = []
    out_data = []
    counts = []
    seen = set()
    idx = -1
    for idx in tqdm.tqdm(range(len(entities))):


        res = requests.post(url_infingram, json={'corpus': 'v4_dolma-v1_6_llama', 'query_type': 'count', 'query': entities[idx]})
        count = res.json()['count']
        time.sleep(1 + np.random.uniform())

        out_ent.append(entities[idx])
        out_data.append(tossup_qs_data[idx])
        counts.append(count)

    tossups['questionArray'] = out_data
    tossups['parsed_text'] = out_ent
    tossups['counts'] = counts

    with open(f'{out_dir}/entity_hard_temp.pkl', 'wb') as handle:
        pickle.dump(tossups, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print(f"Error at initial => Request")
    out.append(None)