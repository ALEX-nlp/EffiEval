# EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization
[![arXiv](https://img.shields.io/badge/arXiv-2508.09662-d63333.svg?logo=arxiv)](https://arxiv.org/abs/2508.09662)

This is the official implementation of **EffiEval**, a training-free benchmarking framework for large language models (LLMs). EffiEval efficiently selects representative subsets of evaluation data, ensuring **representativeness**, **fairness**, and **generalizability** while maintaining strong ranking consistency with full-dataset evaluation. It is scalable and flexible, allowing users to balance evaluation efficiency and reliability. This work is built upon [Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric](https://www.arxiv.org/abs/2504.07440).

![](assets/figures1.gif)
## 1. Installation
Clone the repository and create environment:
```sh
git clone https://github.com/Castria-cn/EffiEval
cd EffiEval
conda create -n effieval python=3.11 -y
conda activate effieval

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
## 2. Usage
### 2.1 Prepare indicator model responses
EffiEval enables efficient benchmarking by first evaluating an indicator model on a given dataset. 
To do this, both the model and dataset need to be prepared. The code structure is organized as follows:
```sh
EffiEval
├── data
│   ├── gsm8k
│   │   └── test.json
│   ├── ...
│   └── mmlu
│       └── test.json
├── get_performance.py # 2.1 Prepare indicator model responses
├── get_neuron.py # 2.2 Compute neurons of the indicator model
├── selection.py # 2.3 Subset selection
└── utils
    ├── dataset.py # dataset configuration
    ├── model.py # model configuration
    └── utils_neuron.py
```
#### 2.1.1 Prepare dataset for evaluation
 In `utils/dataset.py`, the following should be implemented:

- `load_local_dataset(task_name: str) -> list[dict[str, str]]`
- `get_input_sample(task_name: str, sample: dict[str, str]) -> tuple[str, str]`
- The corresponding evaluation function, registered in `EVALUATION_FUNC`
Several examples are provided in the file.

#### 2.1.2 Prepare model for evaluation

In `utils/model.py`, implement the following if necessary:

- `format_tokens`
- `get_model_output`

Then, register the model name in `MODEL_PATHS` (e.g. `"qwen2.5": "Qwen/Qwen2.5-7B-Instruct"`). Several examples are also provided in the file.

> **Note:** When the model is based on an online API, the `MODEL_PATHS` entry should be like:  
> `"gpt-4o-2024-11-20": None`  
> In this case, the preparation steps above can be skipped.
> The `OPENAI_KEY` can be configured in `.env` file.

#### 2.1.3 Run `get_performance.py` to evaluate the model

Example usage:

```python
if __name__ == '__main__':
    get_performance("qwen2.5", "gsm8k")
```

The evaluation results will be saved in the `./response` directory.  
The model name (e.g., `"qwen2.5"`) and dataset name (e.g., `"gsm8k"`) should match the entries in `MODEL_PATHS` and `load_local_dataset`.

---

### 2.2 Compute neurons of the indicator model
When we obtain the outputs of the indicator model, we can compute the neurons activated by each sample based on these outputs. This functionality is implemented in `get_neuron.py`:

```python
if __name__ == '__main__':
    get_neuron("qwen2.5", "gsm8k")
```

The activated neurons of the indicator model on the dataset will be saved in `./neurons` by default.

---

### 2.3 Select a subset using the indicator model
In `selection.py`, load the activated neurons first:

```python
# (indicator_model, topk, dataset_name)
neuron_config = NeuronConfig("qwen2.5", 0.001, "gsm8k")
# np.ndarray with shape [num_sample, num_neuron]
matrix = neuron_config.get_matrix()
```

Then this matrix can then be used to solve the Maximum Coverage Problem (MCP):

```python
indices, coverage = greedy_maximum_coverage(matrix, k=100)
```

- `indices`: `np.ndarray`, indices of the selected samples  
- `coverage`: `int`, number of covered activated neurons

Save the subset to disk:
```python
dataset = load_local_dataset("gsm8k")
subset = [dataset[idx] for idx in indices]

with open("subset.json", "w") as fp:
    json.dump(subset, fp)
```
---

### 2.4 Verify the selected subset

You can verify the subset using `verify_selection` in `selection.py`.  
For example, after evaluating several models (registered in `MODEL_PATHS`) through `get_performance.py`, run:

```python
verify_selection(
    models=list(MODEL_PATHS.keys()),
    task="gsm8k",
    k=100,
    neuron_config=NeuronConfig("qwen2.5", 0.001, "gsm8k")
)
```

This will print the correlation (`r_S`, `r_K`) and MAE between the performance of the models on the full dataset and the selected subset.
## 3. Citation
If you find this work helpful, please consider citing:
```
@misc{cao2025effievalefficientgeneralizable,
      title={EffiEval: Efficient and Generalizable Model Evaluation via Capability Coverage Maximization},
      author={Yixin Cao and Jiahao Ying and Yubo Ma and Yugang Jiang and Yaoning Wang},
      year={2025},
      eprint={2508.09662},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.09662},
}
```