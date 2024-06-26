import pandas as pd
from check_equiv import is_equivalent
import json, os

xlsx = pd.ExcelFile("analysis/All Model Check.xlsx")
TAB_NAMES = [
    "Yi-34B-Chat",
    "claude-3-sonnet-20240229",
    "command-r-plus",
    "claude-3-haiku-20240307",
    "command-r",
    "gpt-4o-2024-05-13",
    "Yi-6B-Chat",
    "Meta-Llama-3-70B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "gpt-4-turbo-2024-04-09",
    "claude-3-opus-20240229",
]

for tab in TAB_NAMES:
    df = pd.read_excel(xlsx, tab)
    # rename columns
    df.columns = ["original_entity", "empty1","generated_question", "empty2", 'generated_entity']
    # drop empty column
    df.drop(columns=['empty1', 'empty2'], inplace=True)
    
    # drop rows with NaN values
    df.dropna(inplace=True)

    # reset index
    df.reset_index(drop=True, inplace=True)

    BREAKS = ['999 maxims', '999', 'Twenty Thousand Leagues Under the Sea', 'Nashville']
    category_idx = 0
    categories = ['num_text', 'num', 'easy_fact', 'hard_fact']

    out_file = f'analysis/{tab}.jsonl'


    if os.path.exists(out_file):
        os.remove(out_file)

    # iterate over rows up to LIMIT
    for i, row in df.iterrows():
        curr_category = categories[category_idx]
        if row['original_entity'] in BREAKS:
            category_idx += 1


        score = is_equivalent(row['original_entity'], row['generated_entity'], curr_category)



        with open(out_file, 'a') as f:
            json.dump({
                'score': score,
                'original_entity': row['original_entity'],
                'generated_entity': row['generated_entity'],
                'category': curr_category,
            }, f)
            f.write('\n')

    break

