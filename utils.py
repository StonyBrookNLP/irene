import os
import json
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from functools import lru_cache
import pickle

import torch
from transformers import AutoConfig
from transformers import AutoModel


def read_json(file_path: str, silent: bool = False):
    """ Reads json to a file """

    if not silent:
        print(f"Reading json from {file_path}")
    with open(file_path, "r") as file:
        dict_obj = json.load(file)
    return dict_obj


def write_json(dict_obj: Dict, file_path: str, silent: bool = False):
    """ Writes json to a file """

    if not silent:
        print(f"Writing json in {file_path}")
    with open(file_path, "w") as file:
        json.dump(dict_obj, file, indent=4)


def parse_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    Parses experiment config from the passed experiment_name using the hardcoded default values.
    Raises error if experiment file is not found, or unexpected config keys are found.
    Prints the parsed config.
    """

    experiment_filepath = f"experiment_configs/{experiment_name}.json"

    assert os.path.exists(experiment_filepath), \
        f"No experiment config found at {experiment_filepath}."

    with open(experiment_filepath, "r") as file:
        experiment_config = json.load(file)

    parsed_config = {}
    parsed_config["ml_experiment_name"] = experiment_config.pop("ml_experiment_name")
    parsed_config["non_ml_experiment_name"] = experiment_config.pop("non_ml_experiment_name", None)
    parsed_config["device_name"] = experiment_config.pop("device_name", "device_1")
    parsed_config["standardize_features"] = experiment_config.pop("standardize_features", True)
    parsed_config["polynomial_features"] = experiment_config.pop("polynomial_features", False)
    parsed_config["normalize_loss_scale"] = experiment_config.pop("normalize_loss_scale", False)
    parsed_config["training_strategy"] = experiment_config.pop("training_strategy", "end2end")
    parsed_config["tanh_scalar"] = experiment_config.pop("tanh_scalar", 10)
    parsed_config["num_epochs"] = experiment_config.pop("num_epochs", 8)
    parsed_config["log_train_diff"] = experiment_config.pop("log_train_diff", False)
    parsed_config["save_nodes"] = experiment_config.pop("save_nodes", False)
    parsed_config["feature_sets"] = experiment_config.pop("feature_sets", ["model_specs", "resource_utilization"])

    if experiment_config:
        assert not experiment_config, \
        f"All experiment configs not used: {experiment_config.keys()}"

    print("Parsed config:")
    print(json.dumps(parsed_config, indent=4))

    return parsed_config


def get_feature_types_to_feature_list(
        feature_types: List[str],
        add_num_parameters: bool = True,
        add_input_size: bool = True
    ) -> List[str]:
    """
    Takes broad feature types and expands it to full feature list.
    """

    feature_types = deepcopy(feature_types)

    feature_list = []
    if "model_specs" in feature_types:
        feature_list.extend([
            "batch_size",
            "seq_len",
            "flops",
            "mem_bytes",
        ])
        feature_types.remove("model_specs")
    if "resource_utilization" in feature_types:
        feature_list.extend([
            "cpu",
            "mem",
            "gpu",
            "gpu_mem",
            "gpu_clk",
            "gpu_mem_clk",
            "times_mean",
            "gpu_energy_mean",
        ])
        feature_types.remove("resource_utilization")
    assert not feature_types, f"feature_types has unrecognized keys {feature_types}"

    if add_input_size:
        feature_list.insert(0, "input_size")

    if add_num_parameters:
        feature_list.insert(0, "num_parameters")

    return feature_list


class Average:
    """
    A simple metric to keep track of an average of values.
    """

    def __init__(self):
        self.value = self.count = 0

    def __call__(self, value: float, count: int = 1) -> None:
        self.value += value
        self.count += count

    def get_value(self, reset: bool=False):
        if reset:
            self.value = self.count = 0
        return float(self.value / self.count) if self.count else 0.0


@lru_cache(maxsize=128)
def get_model(model_name: str) -> AutoModel:
    """
    Takes HF model name and returns pytorch transformer model.
    """

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(config).eval()
    return model


@lru_cache(maxsize=128)
def get_num_parameters(model_name: str, level_name: str) -> int:
    """
    Gets number of parameters in the transformer model at the given level_name.
    """

    model = get_model(model_name)

    num_parameters = 0
    for name, module in model.named_modules():
        if not name:
            continue

        # TODO: There is an error here. For root_nodes the level_name passed isn't right.
        # It's fixed in the main branch.
        # if level_name == model_name:
        #     return sum([parameter.numel() for parameter in model.parameters()])

        if level_name == f'{name}:{module.__class__.__name__}':
            num_parameters = sum([parameter.numel()
                                  for parameter in module.parameters()])
    return num_parameters


def get_percentage_error_item(gold_value: float, predicted_value: float) -> float:
    """Compute percentage error from a gold and predicted value"""
    gold_value = float(gold_value)
    predicted_value = float(predicted_value)
    return 100*abs(gold_value-predicted_value)/gold_value


def get_percentage_error_list(gold_values: List[float], predicted_values: List[float]):
    """Compute percentage error from lists of gold and predicted values"""
    gold_values = [float(e) for e in list(gold_values)]
    predicted_values = [float(e) for e in list(predicted_values)]
    total_error = [get_percentage_error_item(gold_value, predicted_value)
                   for gold_value, predicted_value in zip(gold_values, predicted_values)]
    return sum(total_error) / len(gold_values)


# Hugging Face model-name to model-class string.
HF_NAME2CLASS = {
    'bert-base-uncased': 'BertModel',
    'distilbert-base-uncased': 'DistilBertModel',
    'roberta-base': 'RobertaModel',
    'gpt2': 'GPT2Model',
    'openai-gpt': 'OpenAIGPTModel',
    'distilgpt2': 'GPT2Model',
    "google/bert_uncased_L-10_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-10_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-10_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-10_H-768_A-12": 'BertModel',
    "google/bert_uncased_L-12_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-12_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-12_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-12_H-768_A-12": 'BertModel',
    "google/bert_uncased_L-2_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-2_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-2_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-2_H-768_A-12": 'BertModel',
    "google/bert_uncased_L-4_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-4_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-4_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-4_H-768_A-12": 'BertModel',
    "google/bert_uncased_L-6_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-6_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-6_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-6_H-768_A-12": 'BertModel',
    "google/bert_uncased_L-8_H-128_A-2": 'BertModel',
    "google/bert_uncased_L-8_H-256_A-4": 'BertModel',
    "google/bert_uncased_L-8_H-512_A-8": 'BertModel',
    "google/bert_uncased_L-8_H-768_A-12": 'BertModel'
}


MODEL_LIST = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "distilgpt2",
    "openai-gpt",
    "gpt2",
]

ML_OPERATIONS_LIST = [
    "Embedding",
    "LayerNorm",
    "Linear",
    "Tanh",
    "Conv1D",
    "Softmax",
    "MatMul"
]
