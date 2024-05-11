from abc import ABC, abstractmethod
import random
import copy
from enums import PromptType

# Abstract base class for implementing zero-shot prompts
class ZeroShotPrompt(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def create_prompt(self, data):
        """Create a zero-shot prompt"""
        pass

# Question Generation
class QuestionGenerationVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = f'Generate a one-sentence question that can only be answered by: "{answer}". The question should not contain the text "{answer}". Please format your output as "Question: [insert generated question]"'
        return prompt

class QuestionGenerationCoT(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = QuestionGenerationVanilla().create_prompt(data)
        prompt += ' Let\'s think step by step.'
        return prompt

# Question Answering
class QuestionAnsweringVanilla(ZeroShotPrompt):

    def create_prompt(self, data):
        question = data['input']
        prompt = f'Generate a short answer to the question: "{question}". The answer should only be a few words long. Please format your output as "Answer: [insert generated answer]"'
        return prompt

class QuestionAnsweringCoT(ZeroShotPrompt):

    def create_prompt(self, data):
        answer = data['input']
        prompt = QuestionAnsweringVanilla().create_prompt(data)
        prompt += ' Let\'s think step by step.'
        return prompt


class PromptFactory:

    def __init__(self):

        self.prompt_type_map = {
            PromptType.qg: QuestionGenerationVanilla,
            PromptType.qa: QuestionAnsweringVanilla,
            PromptType.qg_cot: QuestionGenerationCoT,
            PromptType.qa_cot: QuestionAnsweringCoT,
        }

    def get_prompt(self, prompt_type):
        if prompt_type in self.prompt_type_map:
            return self.prompt_type_map[prompt_type]()
        else:
            raise ValueError(f"Unsupported Prompt type: {prompt_type}")
