from abc import ABC, abstractmethod
import random
import copy

from transformers import pipeline, AutoTokenizer
import transformers
import torch
from huggingface_hub.hf_api import HfFolder
from enums import ModelType

# Abstract base class for implementing zero-shot prompts
class LLM(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def generate_text(self, prompt):
        """Generate text from a prompt"""
        pass

class HuggingFaceChatModel(LLM):

    def __init__(self, hf_model_name, temp, min_length, max_length, load_in_4bit, load_in_8bit, device_map, cache_dir, hf_token):

        self.temp = temp
        self.min_length = min_length
        self.max_length = max_length

        HfFolder.save_token(hf_token)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, cache_dir = cache_dir)
        dtype = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "auto": "auto",
        }['auto']
        pipe = pipeline(
            'text-generation',
            model=hf_model_name,
            tokenizer=tokenizer,
            device_map=device_map,
            torch_dtype=dtype,
            min_new_tokens=min_length,
            max_new_tokens=max_length,
            model_kwargs={"cache_dir": cache_dir, "do_sample": False, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
        )
        self.pipe = pipe
        self.tokenizer = tokenizer

    def generate_text(self, prompt):

        messages = [{"role": "user", "content": prompt}]

        if self.temp == 0.0:
            return self.pipe(messages, 
            do_sample=False,  
            min_new_tokens=self.min_length, 
            max_new_tokens=self.max_length,
            return_full_text=False)[0]['generated_text'].strip()
        else:
            return self.pipe(messages, 
            do_sample=True, 
            temperature=self.temp, 
            min_new_tokens=self.min_length, 
            max_new_tokens=self.max_length,
            return_full_text=False)[0]['generated_text'].strip()
    
class ModelFactory:

    @staticmethod
    def get_model(args):
        if args.model_type[0] == ModelType.hf_chat:
            return HuggingFaceChatModel(args.model_name, args.temperature, args.min_tokens, args.max_tokens, args.load_in_4bit, args.load_in_8bit, args.device_map, args.cache_dir, args.hf_token)
        else:
            raise ValueError(f"Unsupported Model type: {args.model_type[0]}")
