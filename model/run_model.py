# imports and directory setup
from data_loader import PromptCollator
from checkpoint_loader import Checkpoint
from model_loader import ModelFactory

import pickle
import datasets
import tqdm
import os
import copy

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
        "--model_nickname",
        '-m',
        type=str,
        help="Nickname of the model in directory",
        default="llama 7b",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model on huggingface/OpenAI",
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model_type",
        type=enum_type(PromptType),
        help="Type of the model: hf_chat",
        default="hf_chat",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Local path or huggingface path pointing to the dataset",
        default="nbalepur/QG_vs_QA_v2",
    )
    parser.add_argument(
        "--inference_split",
        type=str,
        help="Split of the dataset to use",
        default="full",
    )
    parser.add_argument(
        "--load_in_8bit",
        action='store_true',
        help="Should we load the model in 8 bit?",
        default=False,
    )
    parser.add_argument(
        "--load_in_4bit",
        action='store_true',
        help="Should we load the model in 4 bit?",
        default=False,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature of the model to use",
        default=0.7,
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        help="Minimum number of tokens to generate",
        default=5,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum number of tokens to generate",
        default=200,
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        help="Number of few-shot examples to use. Only zero-shot is implemented right now.",
        default=0,
    )
    parser.add_argument(
        "--device_map",
        type=str,
        help="Where to load the model ('cuda', 'auto', 'cpu')",
        default="auto",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Huggingface read token for access to models/datasets",
        default="",
    )
    parser.add_argument(
        "--open_ai_token",
        type=str,
        help="OpenAI token for access to the model",
        default="",
    )
    parser.add_argument(
        '--prompt_types', 
        nargs='*', 
        type=enum_type(PromptType), 
        help='Prompt types/experiments to run', 
        default=[]
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Which partition should be done",
        default="none",
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

    HfFolder.save_token(args.hf_token)

    args = parser.parse_args()
    print(args)
    return args

def main(args):

    # load model
    model_factory = ModelFactory()
    model = model_factory.get_model(args)

    # load checkpoints
    checkpoint_loader = Checkpoint(args)
    start, end = checkpoint_loader.get_bounds()
    
    # load prompt collator
    prompt_collator = PromptCollator(args)

    for pt in args.prompt_types[0]:

        # set output directories
        checkpoint_loader.set_directories(pt) 

        # get prompts and load current save state
        prompts = prompt_collator.get_prompts(pt, checkpoint_loader)
        outputs = checkpoint_loader.load_checkpoint()

        # inference
        for idx in tqdm.tqdm(range(start, end)):
            prompt = prompts[idx]
            out_text = model.generate_text(prompt)
            outputs['raw_text'].append(out_text)
            outputs['prompt'].append(prompt)
            checkpoint_loader.save_checkpoint(outputs, False)

        # final save
        checkpoint_loader.save_checkpoint(outputs, True)

if __name__ == '__main__':
    
    # set up arguments
    args = setup()

    # run inference
    main(args)