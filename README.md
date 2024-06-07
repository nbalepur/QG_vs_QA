# QG vs QA

This repository is the official implementation of "Are Foresight \textit{and} Hindsight 20/20?\\ Exploring Question Generation and Answering Inconsistencies in LLMs", which was our course project for CMSC 828J: Common-sense Reasoning and Natural Language Understanding

## Overview

This repository contains the code and dataset to run our three-step pipeline of:
1. Asking an LLM to generate a question for an entity
2. Asking an LLM to answer its own generated question
3. Comparing whether the LLM's answer matches the original entity

## Datasets

Our datasets contains 3450 factual entities across four diverse categories, and can be viewed on Huggingface [here](https://huggingface.co/datasets/nbalepur/QG_vs_QA_v2) 

## Setup

Python 3.10.0 and pip 23.2.1 were used when running the code in this repository. A list of requirements can be found in `requirements.txt`, which can be installed through the following command:
```
pip install -r requirements.txt 
```

The files in this repository are organized as follows:
* `/data/`: Contains the code to re-collect our dataset
* `/model/`: Contains the code to run the three-step prompting pipeline
* `/analysis/`: Contains the code to run the analysis (just answer equivalence right now)
* `/results/`: Our cached results for our datasets across LLMs
* `/sample_scripts/`: Scripts to easily run the inference code in `/model/`

## Usage

You can run inference on the Huggingface models with the following command: 
```bash
bash .../scripts/model.py
```
You can change the following parameters for each run:
* `model_nickname`: (Nick)name of the model for saving the results. String type
* `model_name`: Name of the model on huggingface. String type
* `model_type`: Type of the model. Currently only supports "hf_chat" (HuggingFace chat models). String type
* `run_name`: Identifier for the run. String type
* `hf_token`: Huggingface read token (for downloading gated models and datasets). String type
*  `device_map`: Device map for the GPUs ("cpu", "cuda", "auto"). String type
*  `partition`: Partition of the dataset. can be "full" or in halves (e.g. "first_half"), quarters (e.g. "first_quarter"), or eigths (e.g. "first_eighth")
*  `experiments`: List of strings denoting experiments to run. Currently supports "qg", "qa", "qg_cot", "qa_cot"

You can also specify the following generation hyperparameters:
* `num_shots`: Number of shots/demonstrations to use. Currently only supports zero-shot prompting. Integer type.
* `temperature`: Model temperature. Float type
* `min_tokens`: Minimum tokens to generate. Integer type
* `max_tokens`: Maximum tokens to generate. Integer type

Finally, the following parameters set up the directories:
* `res_dir`: Pointing to the folder where results are stored
* `prompt_dir`: Pointing to the folder where prompts are stored (can be ignored for zero-shot prompts)
* `cache_dir`: Pointing to the folder where model and dataset downloads can be cached through huggingface

## Running the Pipeline

To run the entire pipeline, complete the following steps:
1. Run `model.sh` using the `qg` flag => Generates questions given the input entity
2. Run `/results/parse_question.py` and specify the run name => Extract the questions from (1)
3. Run `model.sh` using the `qa` flag => Generates an answer for the extracted questions from (2)
4. Run `/results/parse_answer.py` and specify the run name => Extract the answers from (3)
5. Run `/analysis/ae.py` and specify the run name => Compare whether the answer from (4) matches the original entity

## Contact

If you have questions, please contact either of the following authors of the repository:
- [Nishant Balepur](mailto:nbalepur@umd.edu)
- [Feng Gu](mailto:fgu1@umd.edu)
