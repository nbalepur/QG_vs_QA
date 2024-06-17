from enum import Enum

class PromptType(Enum):
    qg = 'qg'
    qa = 'qa'
    qg_cot = 'qg_cot'
    qa_cot = 'qa_cot'

class ModelType(Enum):
    hf_chat = 'hf_chat'
    open_ai = 'open_ai'
    cohere = 'cohere'
    anthropic = 'anthropic'