import os
import json
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from functools import lru_cache
import pickle

import torch


def read_json(file_path: str, silent: bool = False):
    """Reads json to a file"""

    if not silent:
        print(f"Reading json from {file_path}")
    with open(file_path, "r") as file:
        dict_obj = json.load(file)
    return dict_obj


def write_json(dict_obj: Dict, file_path: str, silent: bool = False):
    """Writes json to a file"""

    if not silent:
        print(f"Writing json in {file_path}")
    with open(file_path, "w") as file:
        json.dump(dict_obj, file, indent=4)


def read_jsonl(file_path: str, silent: bool = False):
    """Reads jsonl"""
    if not silent:
        print(f"Reading jsonl from {file_path}")
    with open(file_path, "r") as file:
        json_dicts = [json.loads(line) for line in file.readlines() if line.strip()]
    return json_dicts


def write_jsonl(json_dicts: Dict, file_path: str, silent: bool = False):
    """Writes jsonl"""
    if not silent:
        print(f"Writing jsonl in {file_path}")
    with open(file_path, "w") as file:
        for json_dict in json_dicts:
            file.write(json.dumps(json_dict) + "\n")


def get_feature_types_to_feature_list(
    feature_types: List[str],
    add_num_parameters: bool = True,
    add_input_size: bool = True,
) -> List[str]:
    """
    Takes broad feature types and expands it to full feature list.
    """

    feature_types = deepcopy(feature_types)

    feature_list = []
    if "model_specs" in feature_types:
        feature_list.extend(
            [
                "batch_size",
                "seq_len",
                "flops",
                "mem_bytes",
            ]
        )
        feature_types.remove("model_specs")
    if "resource_utilization" in feature_types:
        feature_list.extend(
            [
                "cpu",
                "mem",
                "gpu",
                "gpu_mem",
                "gpu_clk",
                "gpu_mem_clk",
                "times_mean",
                "gpu_energy_mean",
            ]
        )
        feature_types.remove("resource_utilization")
    assert not feature_types, f"feature_types has unrecognized keys {feature_types}"

    if add_input_size:
        feature_list.insert(0, "input_size")

    if add_num_parameters:
        feature_list.insert(0, "num_parameters")

    return feature_list


def get_percentage_error_item(gold_value: float, predicted_value: float) -> float:
    """Compute percentage error from a gold and predicted value"""
    gold_value = float(gold_value)
    predicted_value = float(predicted_value)
    return 100 * abs(gold_value - predicted_value) / gold_value


def get_percentage_error_list(gold_values: List[float], predicted_values: List[float]):
    """Compute percentage error from lists of gold and predicted values"""
    gold_values = [float(e) for e in list(gold_values)]
    predicted_values = [float(e) for e in list(predicted_values)]
    total_error = [
        get_percentage_error_item(gold_value, predicted_value)
        for gold_value, predicted_value in zip(gold_values, predicted_values)
    ]
    return sum(total_error) / len(gold_values)


def save_pickle(pickle_object: Any, pickle_path: str) -> None:
    """Saves pickle object"""
    with open(pickle_path, "wb") as file:
        pickle.dump(pickle_object, file)


def load_pickle(pickle_path: str) -> Any:
    """Loads pickle object"""
    with open(pickle_path, "rb") as file:
        pickled_object = pickle.load(file)
    return pickled_object


def load_ml_level_model_transformations(model_directory: str) -> Tuple:
    model_filepath = os.path.join(model_directory, "ml_level_model.pkl")
    transformations_filepath = os.path.join(
        model_directory, "ml_level_transformations.pkl"
    )
    model = load_pickle(model_filepath)
    transformations = load_pickle(transformations_filepath)
    return model, transformations


def save_ml_level_model_transformations(
    operationwise_model: Dict, operationwise_transformations: Dict, model_directory: str
) -> Tuple:
    model_filepath = os.path.join(model_directory, "ml_level_model.pkl")
    transformations_filepath = os.path.join(
        model_directory, "ml_level_transformations.pkl"
    )
    save_pickle(operationwise_model, model_filepath)
    save_pickle(operationwise_transformations, transformations_filepath)


def load_non_ml_level_model_transformations(model_directory: str) -> Tuple:
    model_filepath = os.path.join(model_directory, "non_ml_level_model.th")
    transformations_filepath = os.path.join(
        model_directory, "non_ml_level_transformations.pkl"
    )
    model = torch.load(model_filepath)
    transformations = load_pickle(transformations_filepath)
    return model, transformations


def save_non_ml_level_model_transformations(
    model: torch.nn.Module, transformations: List, model_directory: str
):
    model_filepath = os.path.join(model_directory, "non_ml_level_model.th")
    transformations_filepath = os.path.join(
        model_directory, "non_ml_level_transformations.pkl"
    )
    torch.save(model, model_filepath)
    save_pickle(transformations, transformations_filepath)


# These are lowercased to make it
ML_OPERATIONS_LIST = [
    "embedding",
    "layernorm",
    "linear",
    "tanh",
    "conv1d",
    "softmax",
    "matmul",
]

# Configs used in final/full version of IrEne Model.
IRENE_CONFIG = {
    "feature_sets": ["model_specs", "resource_utilization"],
    "polynomial_features": True,
    "training_strategy": "end2end",  # Options: "end2end", "stepwise", "unstructured", "none"
    "standardize_features": True,
    "tanh_scalar": 10,
    "normalize_loss_scale": True,
    "weighting_setting": "close_to_one",  # Options: "exact_one", "close_to_one", "free"
}
