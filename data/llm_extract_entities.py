# imports and directory setup
import pickle
import datasets
import json
import codecs
from transformers import pipeline
import torch
from transformers import AutoTokenizer
import transformers
import torch
import tqdm
import os
import copy
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder

# =========================================== Argument Setup ===========================================

def setup():

    def enum_type(enum):
        enum_members = {e.name: e for e in enum}

        def converter(input):
            out = []
            for x in input.split():
                if x in enum_members:
                    out.append(enum_members[x])
                else:
                    raise argparse.ArgumentTypeError(f"You used {x}, but value must be one of {', '.join(enum_members.keys())}")
            return out

        return converter

    # hyperparameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        '-m',
        type=str,
        help="(Nick)name of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--hf_model_name",
        type=str,
        help="Name of the model on hugging face",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=str, 
        help='Prompt types/experiments to run', 
        default=[]
    )
    parser.add_argument(
        "--train_dataset_split",
        type=str,
        help="Training dataset split",
        default="",
    )
    parser.add_argument(
        "--eval_dataset_split",
        type=str,
        help="Evaluation dataset split",
        default="",
    )
    parser.add_argument(
        "--claim_col",
        type=str,
        help="Column containing the claim",
        default="",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        help="Name of the dataset on huggingface",
        default="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Nickname of the dataset",
        default="",
    )
    parser.add_argument(
        "--stop_token",
        type=str,
        help="When should we stop generating?",
        default="\nClaim:",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="How many new tokens should we generate?",
        default=200,
    )
    parser.add_argument(
        "--load_in_8bit",
        type=str,
        help="Should we load the model in 8 bit?",
        default="False",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=str,
        help="Should we load the model in 4 bit?",
        default="False",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface token for access to the model",
        default="",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Which partition should be done",
        default="full",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        help="Absolute directory of the prompt folder",
        default="./",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Absolute directory of the cache folder for models",
        default="./",
    )
    parser.add_argument(
        "--res_dir",
        type=str,
        help="Absolute directory of the output results folder",
        default="./",
    )

    args = parser.parse_args()
    return args

# =========================================== Load Model ===========================================

def load_model(args):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name, cache_dir=args.cache_dir)

    # set up pipeline
    dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "auto": "auto",
    }['auto']
    pipe = pipeline(
        model=args.hf_model_name,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
        min_new_tokens=5,
        max_new_tokens=args.max_tokens,
        model_kwargs={"cache_dir": args.cache_dir, "do_sample": False, "load_in_4bit": args.load_in_4bit == "True", "load_in_8bit": args.load_in_8bit == "True"}
    )
    return pipe, tokenizer

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen

def generate_text(prompt, stop_token):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
    return pipe(prompt, 
    stopping_criteria=stopping_criteria,
    do_sample=False,
    return_full_text=False)[0]['generated_text'][:-len(stop_token)].strip()

def load_checkpoint(partition, results_dir, pt, start, input_data):

    curr_final = f'{results_dir}/{pt}.pkl'
    if partition != 'full':
        curr_dir = f'{results_dir}/{pt}_{partition}_temp.pkl'
        curr_final = f'{results_dir}/{pt}_{partition}.pkl'
    else:
        curr_dir = f'{results_dir}/{pt}_temp.pkl'
        
    # check final dir
    if os.path.exists(curr_final):
        with open(curr_final, 'rb') as handle:
            answers = pickle.load(handle)
            return answers, start + len(answers)

    # check if no temp dir
    if not os.path.exists(curr_dir):
        return input_data, start
    
    # in progress run
    with open(curr_dir, 'rb') as handle:
        answers = pickle.load(handle)

    return answers, start + len(answers['raw_text'])

def save_checkpoint(partition, results_dir, pt, answers, is_final):
    final_suffix = '' if is_final else '_temp'
    if partition != 'full':
        final_file = f'{results_dir}/{pt}_{partition}{final_suffix}.pkl'
    else:
        final_file = f'{results_dir}/{pt}{final_suffix}.pkl'

    folder_path = '/'.join(final_file.split('/')[:-1])
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    with open(final_file, 'wb') as handle:
        pickle.dump(answers, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_inference(args, pipe, tokenizer):

    # load data
    # if args.hf_dataset_name[0] != '/':
    #     ds = datasets.load_dataset(args.hf_dataset_name)
    #     if len(args.eval_dataset_split) > 0:
    #         ds = ds[args.eval_dataset_split]
    # else:
    #     ds = datasets.load_from_disk(args.hf_dataset_name)
    #     if len(args.eval_dataset_split) > 0:
    #         ds = ds[args.eval_dataset_split]

    # results directory setup
    results_dir = f'{args.res_dir}'
    
    for pt in ['numerical_entity']:
        stop_token = codecs.decode(args.stop_token, 'unicode_escape')

        # load data
        with open(args.prompt_dir, 'rb') as handle:
            input_data = pickle.load(handle)

        input_prompts = input_data
        partition_map = {'full': (0, len(input_data)),
                            'first_half': (0, int(0.5 * len(input_prompts))),
                            'second_half': (int(0.5 * len(input_prompts)), len(input_prompts)),
                            'first_quarter': (0, int(0.25 * len(input_prompts))),
                            'second_quarter': (int(0.25 * len(input_prompts)), int(0.5 * len(input_prompts))),
                            'third_quarter': (int(0.5 * len(input_prompts)), int(0.75 * len(input_prompts))),
                            'fourth_quarter': (int(0.75 * len(input_prompts)), len(input_prompts)),
                            'first_eighth': (0, int(0.125 * len(input_prompts))),
                            'second_eighth': (int(0.125 * len(input_prompts)), int(2*0.125 * len(input_prompts))),
                            'third_eighth': (int(2*0.125 * len(input_prompts)), int(3*0.125 * len(input_prompts))),
                            'fourth_eighth': (int(3*0.125 * len(input_prompts)), int(4*0.125 * len(input_prompts))),
                            'fifth_eighth': (int(4*0.125 * len(input_prompts)), int(5*0.125 * len(input_prompts))),
                            'sixth_eighth': (int(5*0.125 * len(input_prompts)), int(6*0.125 * len(input_prompts))),
                            'seventh_eighth': (int(6*0.125 * len(input_prompts)), int(7*0.125 * len(input_prompts))),
                            'eighth_eighth': (int(7*0.125 * len(input_prompts)), len(input_prompts)),
                            }
        start, end = partition_map[args.partition]
        answers, start = load_checkpoint(args.partition, results_dir, pt, start, input_data)

        # run generation
        for i in tqdm.tqdm(range(start, end)):
            d = answers[i] # get prompts
            if 'raw_text' not in d:
                d['raw_text'] = []
            for prompt in d['prompts']:
                out_text = generate_text(prompt, stop_token) # generate output
                d['raw_text'].append(out_text)
            answers[i] = d
            save_checkpoint(args.partition, results_dir, pt, answers, False)

        save_checkpoint(args.partition, results_dir, pt, answers, True)

if __name__ == '__main__':
    
    # set up arguments
    args = setup()

    # get the model
    pipe, tokenizer = load_model(args)

    # run inference
    run_inference(args, pipe, tokenizer)