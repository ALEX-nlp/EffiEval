"""
To add a new dataset, implement:
1. `load_local_dataset(task_name: str) -> list[dict[str, str]]`:
   Returns a list of data samples, where each sample is a dict.
2. `get_input_sample(task_name: str, sample: dict[str, str]) -> tuple[str, str]`:
   Returns a tuple (system_prompt, prompt).
3. An evaluation function (registered in `EVALUATION_FUNC`):
   Each entry maps str -> callable, where the function takes
   (response, sample) and returns a boolean value.
"""
import json
from typing import Callable

from utils.utils_neuron import (extract_choice, extract_choice_hellaswag,
                                is_correct)

DATASET_PATHS = {
    "gsm8k": "data/gsm8k",
    "arc": "data/arc",
    "mmlu": "data/mmlu",
    "hellaswag": "data/hellaswag",
}

def load_local_dataset(task: str) -> list[dict]:
    assert task in DATASET_PATHS

    if task in ["gsm8k", "arc", "mmlu"]:
        dataset = json.load(open(DATASET_PATHS[task]  + "/" + "test.json", encoding='utf-8'))
    elif task in ["hellaswag"]:
        dataset = json.load(open(DATASET_PATHS[task]  + "/" + "validation.json", encoding='utf-8'))
    else:
        dataset = None
    
    return dataset

def get_input_sample(task: str, sample: dict) -> tuple[str, str]:
    if task == "gsm8k":
        prompt = ""
        question_sample = sample["question"]
        return prompt, question_sample
    elif task == "arc":
        question_sample = sample["question"] + "\n"
        choices = sample["label"]
        for choice in choices:
            question_sample += f'{choice}. {sample[f"{choice}"]}\n'
        return "", question_sample
    elif task == "mmlu":
        prompt = "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        question_sample = sample["question"] + "\n"
        for choice in ["A", "B", "C", "D"]:
            question_sample += f'{choice}. {sample[f"{choice}"]}\n'
        return prompt, question_sample
    elif task == "hellaswag":
        """
        note that in hellaswag community, input is 10-shot by default(llama, qwen); Here we simply adapt zero-shot.
        """
        prompt = "You are given a situation followed by four possible endings. Choose the most appropriate ending by selecting the corresponding number. Respond only with the number of the correct answer.\n\nContext: {}\n"
        question_sample = prompt.format(sample["ctx"]) + "\n"
        for i, choice in enumerate(["1", "2", "3", "4"]):
            question_sample += '{}. {}\n'.format(choice, sample["endings"][i])
        return prompt, question_sample
    
    return None, None

def gsm8k_eval(response, sample) -> bool:
    return is_correct(response, sample["answer"])

def arc_eval(response, sample) -> bool:
    pred = extract_choice(response, [sample[choice] for choice in sample["label"]], sample["label"])
    return pred == sample["answer"]

def mmlu_eval(response, sample) -> bool:
    pred = extract_choice(response, [sample[choice] for choice in ["A", "B", "C", "D"]])
    return pred == sample["answer"]

def hellaswag_eval(response, sample) -> bool:
    pred = extract_choice_hellaswag(response, sample["endings"])
    return int(pred) == int(sample["label"]) + 1

EVALUATION_FUNC: dict[str, Callable[[str, str], bool]] = {
    "gsm8k": gsm8k_eval,
    "arc": arc_eval,
    "mmlu": mmlu_eval,
    "hellaswag": hellaswag_eval,
}