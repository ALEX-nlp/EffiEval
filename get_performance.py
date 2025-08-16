"""
This file is adapted from project MUI-Eval (https://github.com/ALEX-nlp/MUI-Eval)
"""
import copy
import json
import os

import jsonlines
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.dataset import (DATASET_PATHS, EVALUATION_FUNC, get_input_sample,
                           load_local_dataset)
from utils.model import MODEL_PATHS, format_tokens, get_model_output
from utils.utils_neuron import write_csv_row

load_dotenv()

def get_performance_local(model_name: str,
                    task: str,
                    batch_size: int=16,
                    response_dir: str="./response",
                    resume: bool=True):
    # some sanity check
    assert model_name in MODEL_PATHS, f"Unknown model: {model_name}, should be one of {list(MODEL_PATHS.keys())}!"
    assert task in DATASET_PATHS, f"Unknown dataset: {task}, should be one of {list(DATASET_PATHS.keys())}!"
    assert load_local_dataset(task) is not None, f"It seems `load_local_dataset` of task {task} is not implemented!"
    
    model_path = MODEL_PATHS[model_name]
    normalized_model_name = model_name.replace("/", "--").lower()

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    print("Load tokenizer successfully")

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("Load config successfully")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        config=model_config,
        device_map="auto",
        trust_remote_code=True
        ).eval()
    print("Load model successfully")

    # remove already-exist output file
    out_dir = f"{response_dir}/{task}/{normalized_model_name}.jsonl"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    if os.path.exists(out_dir) and not resume:
        os.remove(out_dir)

    torch.cuda.empty_cache()

    if tokenizer:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
   
    # load data
    acc = 0
    file = load_local_dataset(task) 

    evaluation_func = EVALUATION_FUNC.get(task, lambda response, sample: False) # when performance is not needed, return a dummy function return false
    if task not in EVALUATION_FUNC:
        print(f"WARNING: You have not implement evaluation function of task {task}. The success value will be set to 0.0 be default.")
    
    all_samples = list()
    if os.path.exists(out_dir):
        # resume context
        with open(out_dir, "r") as fp:
            for line in fp:
                if len(line.strip()) == 0:
                    continue
                all_samples.append(json.loads(line.strip()))
                acc += all_samples[-1]["success"]

    for i in tqdm(range(0, len(file), batch_size), total=len(file) // batch_size, desc=f"Evaluating dataset {task}..."):
        start_idx = i 
        end_idx = min(i + batch_size, len(file))
        if resume and end_idx <= len(all_samples):
            continue
        batch_texts = file[start_idx:end_idx]
        batch_sample_input = [get_input_sample(task, sample) for sample in batch_texts]
        if "gpt" not in normalized_model_name:
            batch_sample = format_tokens(batch_sample_input, normalized_model_name, tokenizer)
        else:
            batch_sample = batch_sample_input

        outputs = get_model_output(model, tokenizer, normalized_model_name, batch_size, batch_sample)
        for index, sample in enumerate(batch_texts):
            response = outputs[index][0]['generated_text']

            new_sample = copy.deepcopy(batch_texts[index])
            new_sample["sys_input"] = batch_sample_input[index][0]
            new_sample["question_input"] = batch_sample_input[index][1]
            new_sample["response"] =  response

            
            try:
                if evaluation_func(response, sample):
                    acc += 1 
                    new_sample["success"] = 1.0
                else:
                    new_sample["success"] = 0.0
            except: # e.g. bad response format
                new_sample["success"] = 0.0
            
            if (not resume) or (i + index >= len(all_samples)):
                with jsonlines.open(out_dir, 'a') as fw:
                    fw.write(new_sample)
            
            all_samples.append(new_sample)
    
    avg_acc = acc / len(file)

    performance_csv = f"{response_dir}/{task}/result.csv"
    if not os.path.exists(performance_csv):
        write_csv_row(['model','acc'], performance_csv)
    write_csv_row([normalized_model_name, avg_acc], performance_csv)

    # remove previous evaluation results
    df = pd.read_csv(performance_csv) 
    df.drop_duplicates(subset=['model'], keep='last', inplace=True)
    df.to_csv(performance_csv, index=False, encoding='utf-8')

def get_performamce_api(model_name: str,
                        task: str,
                        response_dir: str="./response",
                        resume: bool=True):
    assert task in DATASET_PATHS, f"Unknown dataset: {task}, should be one of {list(DATASET_PATHS.keys())}!"
    assert load_local_dataset(task) is not None, f"`load_local_dataset` of task {task} is not implemented!"
    
    normalized_model_name = model_name.replace("/", "--").lower()

    # remove already-exist output file
    out_dir = f"{response_dir}/{task}/{normalized_model_name}.jsonl"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    if os.path.exists(out_dir) and not resume:
        os.remove(out_dir)
   
    # load data
    acc = 0
    file = load_local_dataset(task) 

    client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
    evaluation_func = EVALUATION_FUNC.get(task, lambda response, sample: False) # when performance is not needed, return a dummy function return false
    if task not in EVALUATION_FUNC:
        print(f"WARNING: You have not implement evaluation function of task {task}. The success value will be set to 0.0 be default.")
    
    all_samples = list()
    if os.path.exists(out_dir):
        # resume context
        with open(out_dir, "r") as fp:
            for line in fp:
                if len(line.strip()) == 0:
                    continue
                all_samples.append(json.loads(line.strip()))
                acc += all_samples[-1]["success"]

    for i, sample in enumerate(tqdm(file)):
        if resume and i < len(all_samples):
            continue
        
        sample_input = get_input_sample(task, sample)

        chat = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": sample_input[1]}],
            temperature=0.0,
            top_p=1.0
        )

        response = chat.choices[0].message.content

        new_sample = copy.deepcopy(sample)
        new_sample["sys_input"] = sample_input[0]
        new_sample["question_input"] = sample_input[1]
        new_sample["response"] =  response
        
        try:
            if evaluation_func(response, sample):
                acc += 1 
                new_sample["success"] = 1.0
            else:
                new_sample["success"] = 0.0
        except: # e.g. bad response format
            new_sample["success"] = 0.0
        
        if (not resume) or (i >= len(all_samples)):
            with jsonlines.open(out_dir, 'a') as fw:
                fw.write(new_sample)
        
        all_samples.append(new_sample)
    
    avg_acc = acc / len(file)

    performance_csv = f"{response_dir}/{task}/result.csv"
    if not os.path.exists(performance_csv):
        write_csv_row(['model','acc'], performance_csv)
    write_csv_row([normalized_model_name, avg_acc], performance_csv)

    # remove previous evaluation results
    df = pd.read_csv(performance_csv) 
    df.drop_duplicates(subset=['model'], keep='last', inplace=True)
    df.to_csv(performance_csv, index=False, encoding='utf-8')

def get_performance(model_name: str,
                    task: str,
                    batch_size: int=16,
                    response_dir: str="./response",
                    resume: bool=True):
    """
    Get performance and output of the model on a certain task.

    Add a new task
    -------
    To add a new task, you should implement: 
    - `get_input_sample` of the corresponding task
    - `load_local_dataset` of the corresponding task -> list[dict], ensure each dict can be processed by `get_input_sample` to form (prompt, question) tuple.
    - `EVALUATION_FUNC`(optional, if performance is needed)

    Add a new model
    -------
    To add a new model, you should implement:
    - `format_tokens`
    - `get_model_output`
    - `NEURON_DICT, NEURON_LAYER_DICT, MODEL_PATHS` in `selection.py` and `utils_neuron.py`

    `get_performance`\\
    |____`format_tokens`(`utils_neuron`)\\
    |____`get_model_output`(`utils_neuron`)\\
    |____`EVALUATION_FUNC`(`utils_neuron`)\\
    |____`load_local_dataset`(`utils_neuron`)\\
    |____`get_input_sample`(this file)

    Parameters
    -------
    - `model_name_or_path`: 
    - `task`:

    Returns
    -------
    None. The response will be saved at `{response_dir}/{task}/{normalized_model_name}.jsonl`, each line a dict with key
    - `sys_input`: `prompt` returned by `get_input_sample`.
    - `question_input`: `question_sample` returned by `get_input_sample`
    - `response`: response of the model.
    - `success`: float, the binary value indicating correctness of this response.
    """
    # some sanity check
    assert model_name in MODEL_PATHS, f"Unknown model: {model_name}, should be one of {list(MODEL_PATHS.keys())}!"
    assert task in DATASET_PATHS, f"Unknown dataset: {task}, should be one of {list(DATASET_PATHS.keys())}!"
    assert load_local_dataset(task) is not None, f"`load_local_dataset` of task {task} is not implemented!"
    
    model_path = MODEL_PATHS[model_name]
    
    if model_path is None: # api
        get_performamce_api(model_name, task, response_dir, resume)
    else:
        get_performance_local(model_name, task, batch_size, response_dir, resume)

if __name__ == '__main__':
    get_performance("qwen2.5", "gsm8k")