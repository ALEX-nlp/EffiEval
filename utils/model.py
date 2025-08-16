from transformers import AutoConfig, TextGenerationPipeline

# For API models, please set its value to None.
MODEL_PATHS = {
    "qwen2.5": "/datacenter/models/Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-2024-11-20": None,
    # ...
}

def format_tokens(samples: list[tuple[str, str]], normalized_model_name: str, tokenizer) -> list[str]:
    return_samples = []
    for sys, question in samples:
        if "llama3" in normalized_model_name or "distill-llama" in normalized_model_name:
            return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
        elif "qwen3" in normalized_model_name:
            messages = [{"role": "user", "content": question}]
            return_sample = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=("think" in normalized_model_name)
            )
        elif "qwen" in normalized_model_name:
            messages = [{"role": "user", "content": question}]
            return_sample = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "llama2" in normalized_model_name:
            return_sample = f'''<s>[INST] {question} [/INST]'''    
        elif "vicuna" in normalized_model_name:
            return_sample = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {question}\nASSISTANT:"
        elif "gemma2" in normalized_model_name:
            messages = messages = [{"role": "user", "content": question}]
            return_sample = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return_samples.append(return_sample)

    return return_samples

def get_model_output(model,
                     tokenizer,
                     normalized_model_name: str,
                     batch_size: int,
                     batch_sample: list,
                     do_sample: bool=False,
                     max_new_tokens: int=1024,
                     return_full_text: bool=False,
                     top_p: float=1.0,
                     temperature: float=0.0) -> list[dict]:
    if "llama3" in normalized_model_name:
        generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, batch_size=batch_size)
        outputs = generator(
            batch_sample, 
            pad_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            do_sample=do_sample, 
            max_new_tokens=max_new_tokens,
            return_full_text=return_full_text, 
            top_p=top_p,
            temperature=temperature
        )
    elif "llama2" in normalized_model_name:
        generator = TextGenerationPipeline(model = model, tokenizer = tokenizer, batch_size=batch_size)
        outputs = generator(
            batch_sample, 
            pad_token_id = tokenizer.eos_token_id,
            do_sample = do_sample, 
            max_new_tokens = max_new_tokens,
            return_full_text = return_full_text, 
            top_p = top_p,
            temperature = temperature
        )
    else:
        generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, batch_size=batch_size)
        outputs = generator(
            batch_sample, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=do_sample, 
            max_new_tokens=max_new_tokens,
            return_full_text=return_full_text, 
            top_p=top_p,
            temperature=temperature
        )

    return outputs

def get_neuron_dict(model_dict: dict[str, str]=None) -> tuple[dict[str, int], dict[str, int]]:
    """
    Load neuron config from model configs.
    """
    if model_dict is None:
        model_dict = MODEL_PATHS
    neuron_dict, neuron_layer_dict = dict(), dict()
    for model in model_dict:
        if model_dict[model] is None:
            continue
        config = AutoConfig.from_pretrained(model_dict[model])
    
        neuron_dict[model] = config.intermediate_size * config.num_hidden_layers
        neuron_layer_dict[model] = config.num_hidden_layers
    
    return neuron_dict, neuron_layer_dict