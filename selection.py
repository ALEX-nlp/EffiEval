import json
import os
import pickle
import random
from collections import defaultdict

import jsonlines
import numpy as np
import orjson
from numba import njit, prange
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from utils.model import MODEL_PATHS, get_neuron_dict
from utils.dataset import load_local_dataset

NEURON_DICT, NEURON_LAYER_DICT = get_neuron_dict()

SAMPLE_T = 5

class NeuronConfig:
    NEURON_DIR = "./neurons"
    CACHE_DIR = "./cache"
    """
    A wrapper to compute neuron of given `(models, threshold, tasks)`. Please make sure `get_performance()` and `get_neuron()` have been called to obtain neurons in `./neurons`

    >>> config = NeuronConfig("llama3.1", 0.001, "arc")
    >>> config.get_matrix() # 2-D 0-1 np.ndarray of shape [num_samples, num_neurons]. For arc num_sample = 1172
    """
    def __init__(self, models: str | list[str], thresholds: list[float] | float, tasks: str | list[str]):
        if not os.path.exists(NeuronConfig.CACHE_DIR):
            os.makedirs(NeuronConfig.CACHE_DIR)

        if isinstance(tasks, str):
            self.tasks = [tasks]
        else:
            self.tasks = tasks
        if isinstance(models, str):
            self.models = [models]
        else:
            self.models = models
        if isinstance(thresholds, float) or isinstance(thresholds, int):
            self.thresholds = [thresholds] * len(self.models)
        else:
            self.thresholds = thresholds
            assert len(self.models) == len(self.thresholds)

    def get_matrix(self):
        if hasattr(self, "matrix"):
            return self.matrix
        matrix_per_task = list()

        for task in self.tasks:
            matrices = list()
            for model, threshold in zip(self.models, self.thresholds):
                matrix = get_activate_neurons(model, task, topk=threshold, return_matrix=True, neuron_dir=NeuronConfig.NEURON_DIR, cache_dir=NeuronConfig.CACHE_DIR)
                matrices.append(matrix)
        
            matrix = np.concatenate(matrices, axis=1) # [num_samples, num_neurons]

            matrix_per_task.append(matrix)

        self.matrix = np.concatenate(matrix_per_task, axis=0) # [num_samples, num_neurons]

        return self.matrix
    
    def __str__(self):
        return "{}_{}_{}".format("_".join(self.models), "_".join(map(str, self.thresholds)), "_".join(self.tasks))

def neuron2id(layer: int, neuron_idx: int) -> int:
    """
    Given `layer` and `neuron_idx`, maps this neuron to an integer.
    """
    if not hasattr(neuron2id, 'lookup'):
        neuron2id.lookup = dict()
    if (layer, neuron_idx) not in neuron2id.lookup:
        neuron2id.lookup[(layer, neuron_idx)] = len(neuron2id.lookup)
    
    return neuron2id.lookup[(layer, neuron_idx)]

def get_activate_neurons(model: str=None,
                         task: str=None,
                         neuron_dir: str="neurons",
                         topk: int|float=0.001,
                         neuron_file: str=None,
                         return_matrix: bool=False,
                         cache_dir: str="./cache") -> set[int] | np.ndarray:
    """
    Given `model` and `tasks`, return a set[int] representing neuron ids(allocated by `neuron2id`) of each task, e.g. {0, 1, 3, ...}

    Parameters
    -------
    - model, task: model and task used to compute neuron
    - neuron_dir: directory to save neurons
    - topk: key neuron threshold(per layer). When `topk` is a `float`, it will be automatically multiplied by (neuron number per layer).
    - neuron_file: when `neuron_file` is not `None`, neuron will be loaded from `neuron_file(.jsonl)`.
    - return_matrix: when set to `True`, a matrix of shape [num_samples, num_neurons] will be returned, representing key neurons (0 / 1) of each sample.
    - cache_dir: (Cache only enabled when `return_matrix` toggled on) the returned matrix object will be cached into `cache_dir`, avoiding repeatedly loading neurons from jsonl file.
    """
    neuron2id.lookup = dict()
    activate_neuron = set()

    assert (model and task) or (neuron_file)

    if return_matrix:
        coverage_matrix = list()
    if model not in NEURON_DICT:
        neuron_file = f"{neuron_dir}/{task}/{model}.jsonl"
        assert os.path.exists(neuron_file)
        model = model.split("_on_")[0]

    if isinstance(topk, float):
        topk = int((NEURON_DICT[model] / NEURON_LAYER_DICT[model]) * topk)
    
    if cache_dir is not None and os.path.exists(f"{cache_dir}/neuron_{model}_{task}_{topk}.pkl") and return_matrix and neuron_file is None:
        print(f"Loading cached neuron matrix from {cache_dir}/neuron_{model}_{task}_{topk}.pkl...")
        with open(f"{cache_dir}/neuron_{model}_{task}_{topk}.pkl", "rb") as fp:
            coverage_matrix = pickle.load(fp)
        return coverage_matrix
    
    normalized_model_name = model.replace("/", "--").lower()
    
    if neuron_file:
        print(f"Opening neuron file {neuron_file}...")
        with open(neuron_file) as fp:
            reader = jsonlines.Reader(fp)
            
            for item in tqdm(reader.iter(), desc=f'Processing task {task}...'): # per case
                neuron_per_case = set()
                top_neurons: list[dict] = item["top_neurons"]
                """
                             |<-- Layer N -->|<-- Layer N-1 -->| ... |<-- Layer 1 -->|
                top_neurons: |High        Low|High          Low| ... |High        Low|
                indices:     |*              |*                |     |*              |
                """
                top_neurons.sort(key=lambda neuron: (neuron["layer"], neuron["score"]), reverse=True)
                indices = np.unique([neuron["layer"] for neuron in top_neurons], return_index=True)[1]

                for idx in indices:
                    top_neuron_layer = [(neuron["layer"], neuron["neuron"]) for neuron in top_neurons[idx: idx+topk]]
                    neuron_per_case.update(map(lambda x: neuron2id(*x), top_neuron_layer))

                activate_neuron.update(neuron_per_case)
                if return_matrix:
                    row = np.zeros((NEURON_DICT[model],))
                    row[list(neuron_per_case)] = 1
                    coverage_matrix.append(row)
    else:
        jsonline_file = f"{neuron_dir}/{task}/{normalized_model_name}.jsonl"

        with open(jsonline_file) as fp:
            for line in tqdm(fp, desc=f'Processing task {task}...'): # per case
                item = orjson.loads(line)
                neuron_per_case = set()
                top_neurons: list[dict] = item["top_neurons"]
                top_neurons.sort(key=lambda neuron: (neuron["layer"], neuron["score"]), reverse=True)
                indices = np.unique([neuron["layer"] for neuron in top_neurons], return_index=True)[1]

                for idx in indices:
                    top_neuron_layer = [(neuron["layer"], neuron["neuron"]) for neuron in top_neurons[idx: idx+topk]]

                    neuron_per_case.update(map(lambda x: neuron2id(*x), top_neuron_layer))

                activate_neuron.update(neuron_per_case)
                if return_matrix:
                    row = np.zeros((NEURON_DICT[model],), dtype=np.int16)
                    row[list(neuron_per_case)] = 1
                    coverage_matrix.append(row)
    
    if return_matrix:
        coverage_matrix = np.stack(coverage_matrix)
        if neuron_file is None and cache_dir is not None:
            with open(f"{cache_dir}/neuron_{model}_{task}_{topk}.pkl", "wb") as fp:
                pickle.dump(coverage_matrix, fp)
        return coverage_matrix

    return activate_neuron

def performance_on_subset(model: str,
                          task: str,
                          k: int=100,
                          response_dir: str="./response",
                          selection: str="effieval",
                          fixed_indices: np.ndarray=None) -> tuple[float, float]:
    success: list[float] = list()
    with open(f"{response_dir}/{task}/{model}.jsonl") as fp:
        for line in fp:
            sample = json.loads(line)
            success.append(sample["success"])
        
    if selection == "effieval":
        indices = fixed_indices
    
    subset_success = np.mean([success[idx] for idx in indices])
    fullset_success = np.mean(success)

    return (subset_success, fullset_success)

@njit(parallel=True)
def compute_cover_num(covered, coverage_matrix, selected_mask):
    n = coverage_matrix.shape[0]
    result = np.zeros(n,) # int32
    for i in prange(n):
        if selected_mask[i]:
            continue
        count = 0
        for j in range(coverage_matrix.shape[1]):
            if (not covered[j]) and coverage_matrix[i, j]: # (covered[j] or coverage_matrix[i, j]):
                count += 1
        result[i] = count
    return result

def greedy_maximum_coverage(
        coverage_matrix: np.ndarray,
        k: int,
        randomize: bool=True,
        verbose: bool=True,
    ) -> np.ndarray:
    """
    Cover the neurons using greedy algorithm.
    Time complexity `O(k * sample_num)`

    Parameters
    -------
    - `coverage_matrix`: a 0-1 matrix of shape [sample_num, neuron_num]. cover_matrix[i, j] == 1 means the sample activates neuron j.(at a specific threshold)

    Returns
    -------
    tuple, (np.ndarray, int)
    `np.ndarray`: the selected indices of the samples. Each element is in [0, sample_num - 1].
    `int`: #(covered neurons)
    """
    coverage_matrix = coverage_matrix.astype(np.bool_)
    
    sample_num, neuron_num = coverage_matrix.shape
    covered = np.zeros((neuron_num, ), dtype=np.bool_)
    selected = list()

    ratios_per_sample = list()
    selected_mask = np.zeros(coverage_matrix.shape[0], dtype=bool)

    for _ in tqdm(range(k), disable=not verbose):
        cover_num = compute_cover_num(covered, coverage_matrix, selected_mask)

        if not randomize:
            selected_idx = np.argmax(cover_num)
        else:
            candidate_indices = np.where(cover_num == np.max(cover_num))[0]

            selected_idx = random.choice(candidate_indices)
        selected.append(selected_idx)
        selected_mask[selected_idx] = True
        covered |= coverage_matrix[selected_idx]

        ratios_per_sample.append(covered.sum() / neuron_num)
    
    if verbose:
        print(f"Coverage of all the {sample_num} samples: {np.any(coverage_matrix, axis=0).sum()} / {neuron_num}({np.any(coverage_matrix, axis=0).sum() / neuron_num})")

    return np.array(selected), covered.sum()

def greedy_maximum_coverage_by_r(
        coverage_matrix: np.ndarray,
        k: float,
        randomize: bool=True,
        return_coverage_rates: bool=False,
    ) -> np.ndarray:
    coverage_matrix = coverage_matrix.astype(np.bool_)
    
    sample_num, neuron_num = coverage_matrix.shape
    covered = np.zeros((neuron_num, ), dtype=np.bool_)
    selected = list()

    ratios_per_sample = list()
    selected_mask = np.zeros(coverage_matrix.shape[0], dtype=bool)
    
    all_covered = np.any(coverage_matrix, axis=0).sum()

    while covered.sum() / all_covered < k:
        cover_num = compute_cover_num(covered, coverage_matrix, selected_mask)

        if not randomize:
            selected_idx = np.argmax(cover_num)
        else:
            candidate_indices = np.where(np.max(cover_num) == cover_num)[0]
            selected_idx = random.choice(candidate_indices)
        selected.append(selected_idx)
        selected_mask[selected_idx] = True
        covered |= coverage_matrix[selected_idx]

        ratios_per_sample.append(covered.sum() / neuron_num)
    
    print(f"Sample number under coverage ratio={k}: {len(selected)}")
    
    print(f"Coverage of all the {sample_num} samples: {np.any(coverage_matrix, axis=0).sum()} / {neuron_num}({np.any(coverage_matrix, axis=0).sum() / neuron_num})")

    if return_coverage_rates:
        return (np.array(selected), (np.any(coverage_matrix, axis=0).sum() / neuron_num), ratios_per_sample)

    return np.array(selected), covered.sum()

def selection_result(file_path: str):
    print(f"Result file: {file_path}")

    file = open(file_path)
    methods = defaultdict(list)
    all_models = set()
    raw_data = list()

    for item in file:
        if len(item.strip()) == 0:
            continue
        model, task, subset, fullset, selection = item.strip().split(",")
        raw_data.append((model, task, subset, fullset, selection))
        selection = selection.strip()
        all_models.add(model)
        methods[selection].append((float(subset), float(fullset)))
    
    print(f"Available {len(all_models)} models in evaluation: {all_models}")

    # calculate pearson corr
    for method in methods:
        points = methods[method]
        subsets = [item[0] for item in points]
        fullsets = [item[1] for item in points]

        total_eval = len(subsets)
        assert total_eval % SAMPLE_T == 0
        model_num = total_eval // SAMPLE_T

        spearman_corr = np.mean([spearmanr(subsets[idx: idx+model_num], fullsets[idx: idx+model_num])[0] for idx in range(0, total_eval, model_num)])
        tau = np.mean([kendalltau(subsets[idx: idx+model_num], fullsets[idx: idx+model_num])[0] for idx in range(0, total_eval, model_num)])

        spearman_std = np.std([spearmanr(subsets[idx: idx+model_num], fullsets[idx: idx+model_num])[0] for idx in range(0, total_eval, model_num)])
        tau_std = np.std([kendalltau(subsets[idx: idx+model_num], fullsets[idx: idx+model_num])[0] for idx in range(0, total_eval, model_num)])

        
        mae = np.abs(np.array(subsets) - np.array(fullsets)).mean()

        print(f"spearman corr of method {method}: {spearman_corr}")
        print(f"Kendall's tau of method {method}: {tau}")
        print(f"Spearman corr std. of method {method}: {spearman_std}")
        print(f"Kendall's tau std. of method {method}: {tau_std}")
        print(f"MAE of method {method}: {mae}")

def get_result_matrix(models: list[str],
                      task: str,
                      result_dir="./response"):
    # shape [num_item, num_models]
    result_matrix = list()
    for model in models:
        model_result = list()
        with open(f"{result_dir}/{task}/{model}.jsonl") as fp:
            for line in fp:
                if line.strip():
                    model_result.append(bool(json.loads(line)["success"]))
        
        result_matrix.append(model_result)
    
    return np.array(result_matrix, dtype=np.bool_).T

def verify_selection(
        models: list[str],
        task: str,
        k: int | float,
        neuron_config: NeuronConfig,
        response_dir: str="./response",
        output_dir: str="./results",
        overwrite: bool=True,
    ):
    """
    Check selection performance by `Spearman corr.`, `Mean Absolute Error(MAE)`.
    All data will be saved in `output_dir`.

    Parameters
    -------
    - `models`: models to evaluate
    - `tasks`: tasks to evaluate on
    - `k`: #sample to retain
    - `neuron_config`: configuration of the indicator model, refer to `NeuronConfig` class in this file
    - `overwrite`: whether to overwrite the temp file
    """
    os.makedirs(output_dir, exist_ok=True)

    txt_dir = output_dir + f"/k={k}_{str(neuron_config)}.txt"
    if overwrite and os.path.exists(txt_dir):
        os.remove(txt_dir)

    coverage_matrices = neuron_config.get_matrix()
    
    for _ in tqdm(range(SAMPLE_T), f"Evaluating EffiEval..."):
        indices, _ = greedy_maximum_coverage(coverage_matrices, k, randomize=True)
        
        for model in models:
            subset, full_set = performance_on_subset(model, task, k=k, selection="effieval", fixed_indices=indices, response_dir=response_dir)

            with open(txt_dir, "a") as fp:
                fp.write(f"{model},{task},{subset},{full_set},effieval\n")
    
    selection_result(txt_dir)

if __name__ == '__main__':
    neuron_config = NeuronConfig("llama3.1", 0.001, "gsm8k")
    matrix = neuron_config.get_matrix()

    indices, _ = greedy_maximum_coverage(matrix, 100)

    dataset = load_local_dataset("gsm8k")
    subset = [dataset[idx] for idx in indices]

    with open("subset.json", "w") as fp:
        json.dump(subset, fp)