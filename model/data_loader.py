from enum import Enum
import datasets
import numpy as np
from prompt import *
import pickle
import contextlib
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from abc import ABC, abstractmethod

class DataFetcher(ABC):
    @abstractmethod
    def get_data(self):
        """Retrieve data from the source."""
        pass

class EntityFetcher(DataFetcher):

    def __init__(self, ds_name, split_name):
        if os.path.isfile(ds_name):
            self.ds = datasets.load_from_disk(ds_name)
        else:
            self.ds = datasets.load_dataset(ds_name)
        if type(self.ds) == datasets.dataset_dict.DatasetDict:
            if split_name in self.ds.keys():
                self.ds = self.ds[split_name]
            else:
                raise ValueError(f"The split does not exist in your dataset dictionary: {split_name}")
            
    def get_data(self, column_name='entity'):
        return list(self.ds[column_name])

class QuestionFetcher(DataFetcher):

    def __init__(self, res_dir):
        if os.path.isfile(res_dir):
            with open(f'{res_dir}', 'rb') as handle:
                self.ds = pickle.load(handle)
        else:
            raise ValueError(f"The question file does not exist: {res_dir}")

    def get_data(self, column_name='question'):
        return self.ds[column_name]

class DataFetcherFactory:

    @staticmethod
    def get_data_fetcher(prompt_type, args, checkpoint_loader):
        if source_type in {PromptType.qg, PromptType.qg_cot}:
            return EntityFetcher(args.dataset_name, args.inference_split)
        elif source_type in {PromptType.qa, PromptType.qa_cot}:
            swapped_dir = checkpoint_loader.get_final_dir().replace('qa', 'qg')
            return QuestionFetcher(swapped_dir)
        else:
            raise ValueError(f"Unsupported DataFetcher type: {prompt_type}")

class PromptCollator:

    def __init__(self, args):
        self.prompt_factory = prompt.PromptFactory()
        self.data_fetcher_factory = DataFetcherFactory()
        self.args = args

    def get_prompts(prompt_type, checkpoint_loader):

        data_fetcher = self.data_fetcher_factory.get_data_fetcher(prompt_type, self.args, checkpoint_loader)
        prompt_parser = self.prompt_factory.get_prompt(prompt_type)

        prompt_inputs = data_fetcher.get_data()
        prompts = [prompt_parser.create_prompt({'input': p}) for p in prompt_inputs]
        return prompts
        