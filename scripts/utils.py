import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from ray.data import DatasetContext
from ray.train.torch import get_device

from scripts.config import mlflow

DatasetContext.get_current().execution_options.preserve_order = True


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_dict(path: str) -> Dict:
    """
    Load Dictionary from JSON file.

    Args:
        path (str): Filepath of the JSON file

    Returns:
        dict: Loaded dictionary
    """
    with open(path) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, path: str, cls: Any = None, sortkeys: bool = False) -> None:
    """
    Save Dictionary to JSON file.

    Args:
        d (dict): Dictionary to be saved
        path (str): Filepath to save the JSON file
        cls (Any, optional): Custom encoder class for JSON serialization
        sortkeys (bool, optional): Whether to sort the keys in the JSON file

    Returns:
        None
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)
        fp.write("\n")


def pad_array(arr: np.ndarray, dtype=np.int32) -> np.ndarray:
    """
    Function to pad a 2D NumPy array with zeros to make all rows have the same length.

    Args:
        arr (np.ndarray): Input 2D NumPy array
        dtype (type, optional): Data type of the padded array. Defaults to np.int32

    Returns:
        np.ndarray: Padded 2D NumPy array
    """
    max_len = max(len(row) for row in arr)
    padded_arr = np.zeros((arr.shape[0], max_len), dtype=dtype)
    for i, row in enumerate(arr):
        padded_arr[i][: len(row)] = row
    return padded_arr


def collate_fn(batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching data, specifically designed for handling arrays
    of varying lengths of tokenized sequences.

    Args:
        batch (Dict[str, np.ndarray]): Input batch dictionary containing arrays

    Returns:
        Dict[str, torch.Tensor]: Batching dictionary with arrays converted to PyTorch tensors
    """
    batch["ids"] = pad_array(batch["ids"])
    batch["masks"] = pad_array(batch["masks"])
    dtypes = {"ids": torch.int32, "masks": torch.int32, "targets": torch.int64}
    tensor_batch = {}
    for key, array in batch.items():
        tensor_batch[key] = torch.as_tensor(array, dtype=dtypes[key], device=get_device())
    return tensor_batch


def get_run_id(experiment_name: str, trial_id: str) -> str:
    """
    Get MLflow Run ID.

    Args:
        experiment_name (str): Name of the MLflow experiment
        trial_id (str): ID of the trial

    Returns:
        str: MLflow Run ID for the specified experiment and trial
    """
    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(experiment_names=[experiment_name], filter_string=f"tags.trial_name = '{trial_name}'").iloc[0]
    return run.run_id


def dict_to_list(data: Dict, keys: List[str]) -> List[Dict[str, Any]]:
    """
    Convert Dictionary to List of Dictionaries.

    Args:
        data (Dict): Input dictionary with lists.
        keys (List[str]): List of keys to include in the output dictionaries

    Returns:
        List[Dict[str, Any]]: List of dictionaries with selected keys
    """
    list_of_dicts = []
    for i in range(len(data[keys[0]])):
        new_dict = {key: data[key][i] for key in keys}
        list_of_dicts.append(new_dict)
    return list_of_dicts
