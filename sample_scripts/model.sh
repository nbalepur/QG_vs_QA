#!/bin/bash
# ... activate your venv here! ...
# ...

# dataset details
inference_split="subset"
dataset_name="nbalepur/QG_vs_QA_v2"

# model details
model_nickname="llama3_70b_chat"
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_type="hf_chat"

# how to identify this run
run_name="0_shot_0_temp"

# model parameters
num_shots=0
temperature=0.0
min_tokens=5
max_tokens=200
hf_token="..." # huggingface token read (for downloading gated models and datasets)
open_ai_token="" # OpenAI token (for GPT models)
device_map="auto" # device map ('cpu', 'cuda', 'auto')
partition="full"  # partition of the dataset. can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")

# list of experiments
# see all possible experiments in: /mcqa-artifacts/model/data_loader.py
experiments=("qg")

# directory setup
res_dir=".../QG_vs_QA/results/" # Results folder directory
prompt_dir=".../QG_vs_QA/prompts" # Prompt folder directory
cache_dir="D:/hf_cache" # Cache directory to save the model

experiments_str=$(IFS=" "; echo "${experiments[*]}")

# add the correct file below
# there are also flags for `load_in_4bit` and `load_in_8bit`
python3 .../QG_vs_QA/model/run_model.py \
--run_name="$run_name" \
--model_nickname="$model_nickname" \
--model_name="$model_name" \
--model_type="$model_type" \
--dataset_name="$dataset_name" \
--inference_split="$inference_split" \
--partition="$partition" \
--hf_token="$hf_token" \
--open_ai_token="$open_ai_token" \
--device_map="$device_map" \
--num_shots="$num_shots" \
--temperature="$temperature" \
--min_tokens="$min_tokens" \
--max_tokens="$max_tokens" \
--prompt_types="$experiments_str" \
--res_dir="$res_dir" \
--prompt_dir="$prompt_dir" \
--cache_dir="$cache_dir"
