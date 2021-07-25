import re
import pickle
import subprocess
from typing import List, Dict
import json
import csv
from copy import deepcopy
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "graph_extractor"))

import common # For monkey patched code.
from visualise_model_as_graph import run_model_to_graph

from utils import (
    write_json, parse_experiment_config, get_num_parameters, Average,
    get_percentage_error_item, get_feature_types_to_feature_list, HF_NAME2CLASS, MODEL_LIST
)
from generate_all_experiment_configs import generate_all_experiment_configs

import warnings
warnings.filterwarnings("ignore")

torch.random.manual_seed(13370)
os.environ["CUDA_VISIBLE_DEVICES"] = "" #GPU makes it slower.


def y2u_id(model_name, instance_type, batch_size, seq_len, scope):
    """
    Hacky alignment: Needs to be updated in camera-ready.
    """

    if instance_type == "softmax":
        scope = re.sub(r'\.softmax$', '', scope)

    if instance_type == "matmul":
        scope = re.sub(r'\.matmul$', '', scope)

    return (model_name, instance_type, batch_size, seq_len, scope)

def q2u_scope(scope, module_list_scope_names):
    """
    Hacky alignment: Needs to be updated in camera-ready.
    """

    return '.'.join([arg for arg in scope.split('.') if arg not in module_list_scope_names])

def q2u_dictionary(dictionary, module_list_scope_names):
    """
    Hacky alignment: Needs to be updated in camera-ready.
    """

    copied_dictionary = {}
    count = 0
    for key, value in dictionary.items():
        scope = key[4] # Delicate, take Care! #TODO: Fix
        scope = q2u_scope(scope, module_list_scope_names)
        key = list(key)
        key[4] = scope
        key = tuple(key)
        copied_dictionary[key] = value
        count += 1
    return copied_dictionary


def get_node_energy(node,
                    leaf_node_energies,
                    model_name,
                    batch_size,
                    seq_len) -> float:
    id_ = y2u_id(
        model_name,
        node.instance_type,
        batch_size,
        seq_len,
        node.scope
    ) # Hacky alignment: Needs to be updated in camera-ready.

    if not node.child_nodes:
        return leaf_node_energies[id_]

    total_energy = 0.0
    for child_node in node.child_nodes:
        energy = get_node_energy(child_node, leaf_node_energies,
                                 model_name, batch_size, seq_len)
        total_energy += energy

    return total_energy

def get_node_from_tree_dict(tree_dict, level_name, module_list_scope_names):
    scope, _ = level_name.split(":")
    scope = q2u_scope(scope, module_list_scope_names)
    node = tree_dict["root."+scope]
    return node


def get_leaf_node_energies(feature_filepaths, model_list, energy_key):

    leaf_node_energies = {}
    for feature_filepath in feature_filepaths:
        with open(feature_filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:

                model_name = row['model_name']

                if model_name not in model_list:
                    continue

                level_name = row['level_name']
                batch_size = int(float(row['batch_size']))
                seq_len = int(float(row['seq_len']))
                scope, instance_type = level_name.split(":")
                scope = "root."+scope

                if instance_type in ["Softmax", "MatMul"]:
                    instance_type = instance_type.lower()

                id_ = (
                    model_name,
                    instance_type,
                    batch_size,
                    seq_len,
                    scope
                )

                energy = float(row[energy_key])
                leaf_node_energies[id_] = energy

    return leaf_node_energies


def get_node_features_and_energies(feature_filepaths, model_list, feature_list,
                                   add_input_size: bool=True, add_num_parameters: bool=True,
                                   standardize_features: bool=True,
                                   polynomial_features: bool=False):
    node_features = {}
    node_energies = {}
    for feature_filepath in feature_filepaths:
        with open(feature_filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                model_name = row['model_name']
                level_name = row['level_name']
                batch_size = int(float(row['batch_size']))
                seq_len = int(float(row['seq_len']))

                if model_name not in model_list:
                    continue

                if ":" in level_name:
                    # Non-root
                    scope, instance_type = level_name.split(":")
                    scope = "root."+scope
                else:
                    # Root
                    instance_type = HF_NAME2CLASS[model_name]
                    scope = "root"

                if instance_type in ["Softmax", "MatMul"]:
                    instance_type = instance_type.lower()

                features = [float(row[key]) for key in feature_list]

                if add_input_size:
                    input_size = float(batch_size)*float(seq_len)
                    features.insert(0, input_size)

                if add_num_parameters:
                    model_name = row['model_name']
                    level_name = row['level_name']
                    num_parameters = get_num_parameters(model_name, level_name)
                    features.insert(0, num_parameters)

                gold_energy = float(row["energy_mean"])

                id_ = (
                    model_name,
                    instance_type,
                    batch_size,
                    seq_len,
                    scope
                )

                node_features[id_] = torch.tensor(features).unsqueeze(0)
                node_energies[id_] = torch.tensor(gold_energy)

    if standardize_features or polynomial_features:
        all_features = []
        idx2id = {}
        for idx, (id_, features) in enumerate(node_features.items()):
            idx2id[idx] = id_
            features = np.array(features)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)

        if standardize_features:
            all_features = StandardScaler().fit_transform(all_features)
            all_features = torch.tensor(all_features)

        if polynomial_features:
            poly_reg = PolynomialFeatures(degree=3)
            all_features = poly_reg.fit_transform(all_features)
            all_features = torch.tensor(all_features, dtype=torch.float32)

        for idx, features in enumerate(all_features):
            node_features[idx2id[idx]] = features.unsqueeze(0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    for key in list(node_features.keys()):
        node_features[key] = node_features[key].to(device)

    for key in list(node_energies.keys()):
        node_energies[key] = node_energies[key].to(device)

    return node_features, node_energies


def feature_filepaths_to_rows(feature_filepaths):
    rows = []
    for feature_filepath in feature_filepaths:
        with open(feature_filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                rows.append(row)
    return rows


class WeightingModel(torch.nn.Module):
    def __init__(self, num_features: int, setting: str, tanh_scalar: int=10):
        super(WeightingModel, self).__init__()
        self.num_features = num_features
        self.module = torch.nn.Linear(num_features, 1)
        assert setting in ("exact_one", "close_to_one", "free"), f"invalid setting {setting}."
        self.setting = setting
        self.tanh_scalar = tanh_scalar

    def forward(self, features):
        if self.setting == "exact_one":
            return torch.tensor(1.0)

        elif self.setting == "close_to_one":
            return 1 + (self.module(features).tanh().squeeze()/self.tanh_scalar)

        elif self.setting == "free":
            return self.module(features).squeeze()


class RecursiveTreeComputation(torch.nn.Module):


    def __init__(
            self,
            num_features: int,
            weighting_setting: str,
            tanh_scalar: float = 10,
            training_strategy: str = "end2end",
            normalize_loss_scale: bool = False
        ):
        super(RecursiveTreeComputation, self).__init__()
        self.weighting_model = WeightingModel(
            num_features=num_features, setting=weighting_setting, tanh_scalar=tanh_scalar
        )
        self.training_strategy = training_strategy
        if training_strategy:
            assert training_strategy in ("end2end", "stepwise", "unstructured", "none"), \
            "invalid train strategy."
        self.normalize_loss_scale = normalize_loss_scale

    def forward(self, node, training: bool=False):

        if not node.child_nodes:
            assert node.predicted_energy is not None, "Found leaf node w/o energy prediction."
            node.loss = node.subtree_loss = 0.0
            node.subtree_error_sum = 0.0
            node.subtree_error_count = 0
            node.subtree_error_perc = 0.0
            return node

        node.predicted_energy = 0

        node.subtree_loss = 0.0
        node.subtree_error_sum = 0.0
        node.subtree_error_count = 0

        for child_node in node.child_nodes:

            child_node = self.forward(child_node, training=training)
            child_node_score = self.weighting_model.forward(child_node.features)

            if self.training_strategy in ["none", "end2end"]:
                node.predicted_energy += (child_node_score*child_node.predicted_energy)

            elif self.training_strategy == "stepwise":
                if training:
                    node.predicted_energy += (child_node_score*child_node.gold_energy)
                else:
                    node.predicted_energy += (child_node_score*child_node.predicted_energy)

            # TODO: Add an option to toggle between using only root node supervision
            # vs all nodes supervision.
            node.subtree_loss += child_node.subtree_loss
            node.subtree_error_sum += child_node.subtree_error_sum
            node.subtree_error_count += child_node.subtree_error_count

        if self.training_strategy == "unstructured":
            node.predicted_energy = self.weighting_model.forward(node.features)

        mse_loss_func = torch.nn.MSELoss()
        node.loss = mse_loss_func(node.predicted_energy, node.gold_energy)
        if self.normalize_loss_scale:
            node.loss = node.loss / (node.gold_energy**2)
        node.subtree_loss += node.loss

        node.subtree_error_sum += get_percentage_error_item(node.gold_energy, node.predicted_energy)
        node.subtree_error_count += 1
        node.subtree_error_perc = float(node.subtree_error_sum/node.subtree_error_count)

        return node


def annotate_tree(node, node_energies, leaf_node_energies, node_features, model_name,
                  batch_size, seq_len):
    id_ = y2u_id(
        model_name,
        node.instance_type,
        batch_size,
        seq_len,
        node.scope
    )
    node.model_name = model_name
    node.batch_size = batch_size
    node.seq_len = seq_len

    node.gold_energy = node_energies.get(id_, torch.tensor(0.0))
    node.features = node_features.get(id_, torch.tensor([0.0]*680).unsqueeze(0))
    # TODO: Remove the default values after QQ fixes the missing batches.

    if id_ not in node_energies:
        print(f"Can't index: {id_}")

    for child_node in node.child_nodes:
        annotate_tree(child_node, node_energies, leaf_node_energies, node_features,
                      model_name, batch_size, seq_len)

    if not node.child_nodes:
        node.predicted_energy = float(leaf_node_energies.get(id_, 0.0))
    return node


def make_tree_serializable(node):

    node.features = node.features.numpy().tolist()
    node.gold_energy = float(node.gold_energy)
    node.predicted_energy = float(node.predicted_energy)

    node.loss = float(node.loss)
    node.subtree_loss = float(node.subtree_loss)

    for child_node in node.child_nodes:
        make_tree_serializable(child_node)


def predict_and_evaluate(recursive_tree, model_list, rows,
                         num_features, node_energies, leaf_node_energies,
                         node_features, use_root, save_annotated_trees_dir=None):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    recursive_tree.to(device)
    predictions = []
    perc_error_sum = 0
    perc_error_count = 0

    root_error_metric = Average()
    tree_error_metric = Average()
    modelwise_root_error_metric = defaultdict(Average)
    modelwise_tree_error_metric = defaultdict(Average)

    print(f"Predicting on {model_list}")
    for row in tqdm(rows):

        batch_size = int(float(row['batch_size']))
        seq_len = int(float(row['seq_len']))
        model_name = row['model_name']
        level_name = row['level_name']

        if model_name not in model_list:
            continue

        gold_energy = float(row["energy_mean"])

        tree_node, tree_dict, module_list_scope_names = run_model_to_graph(model_name, device)
        u_node_energies = q2u_dictionary(node_energies, module_list_scope_names)
        u_node_features = q2u_dictionary(node_features, module_list_scope_names)
        u_leaf_node_energies = q2u_dictionary(leaf_node_energies, module_list_scope_names)

        if not use_root:
            tree_node = get_node_from_tree_dict(tree_dict, level_name, module_list_scope_names)

        tree_node = annotate_tree(tree_node, u_node_energies, u_leaf_node_energies,
                                  u_node_features, model_name, batch_size, seq_len)
        tree_node = recursive_tree(tree_node, training=False)

        make_tree_serializable(tree_node)

        tree_error_metric(tree_node.subtree_error_sum, tree_node.subtree_error_count)
        modelwise_tree_error_metric[model_name](tree_node.subtree_error_sum, tree_node.subtree_error_count)

        root_error = get_percentage_error_item(tree_node.gold_energy, tree_node.predicted_energy)
        root_error_metric(root_error)
        modelwise_root_error_metric[model_name](root_error)

        if save_annotated_trees_dir:
            # Assure some annotations are present.
            assert tree_node.predicted_energy is not None
            assert tree_node.child_nodes[0].predicted_energy is not None

            os.makedirs(save_annotated_trees_dir, exist_ok=True)
            filename = f"model_root__{model_name}__{batch_size}__{seq_len}"
            filename = filename.replace('/', '_')

            # filepath = os.path.join(save_annotated_trees_dir, f"{filename}.pkl")
            # with open(filepath, "wb") as pkl_file:
            #     pickle.dump(tree_node, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

            from serialize_to_json import serialize_to_json
            tree_node_json = serialize_to_json(tree_node)
            filepath = os.path.join(save_annotated_trees_dir, f"{filename}.json")
            with open(filepath, "w") as file:
                json.dump(tree_node_json, file)

        assert round(tree_node.gold_energy, 1) == round(gold_energy, 1)

        predictions.append({
            "true_ys": tree_node.gold_energy,
            "pred_ys": tree_node.predicted_energy
        })

    result = {
        "root_percentage_error": round(root_error_metric.get_value(), 1),
        "modelwise_percentage_error": {
            key: round(metric.get_value(), 1)
            for key, metric in modelwise_root_error_metric.items()
        },
        "predictions": predictions
    }

    if use_root:
    # If we don't use roots (models), the modules will duplicate many times.
    # So node_count or perc_error_sum won't be correct.
        result["node_percentage_eror"] = round(tree_error_metric.get_value(), 1)
        result["modelwise_node_percentage_eror"] = {
            key: round(metric.get_value(), 1)
            for key, metric in modelwise_tree_error_metric.items()
        }

    return result


def train_weighting_model(recursive_tree, model_list, rows, num_features,
                          node_energies, leaf_node_energies, node_features, num_epochs):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    recursive_tree = recursive_tree.to(device)

    optimizer = torch.optim.Adam(recursive_tree.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(recursive_tree.parameters(), lr=1e-3, momentum=0.9)

    tqdm_obj = tqdm(range(num_epochs))
    for _ in tqdm_obj:

        total_loss = 0.0
        perc_error_sum = 0.0
        perc_error_count = 0
        for row in rows:

            batch_size = int(float(row['batch_size']))
            seq_len = int(float(row['seq_len']))
            model_name = row['model_name']
            level_name = row['level_name']

            if model_name not in model_list:
                continue

            root_node, _, module_list_scope_names = run_model_to_graph(model_name, device)
            u_node_energies = q2u_dictionary(node_energies, module_list_scope_names)
            u_node_features = q2u_dictionary(node_features, module_list_scope_names)
            u_leaf_node_energies = q2u_dictionary(leaf_node_energies, module_list_scope_names)

            root_node = annotate_tree(root_node, u_node_energies, u_leaf_node_energies,
                                      u_node_features, model_name, batch_size, seq_len)
            root_node = recursive_tree(root_node, training=True)

            loss = root_node.subtree_loss
            assert loss.requires_grad
            total_loss += loss.item()

            perc_error_sum += float(root_node.subtree_error_sum)
            perc_error_count += int(root_node.subtree_error_count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        perc_error_mean = round(perc_error_sum / perc_error_count, 6)
        loss_mean = round(float(total_loss / perc_error_count), 6)

        tqdm_obj.set_description(f"Mean Node Loss: {loss_mean} Mean Node %Error: {perc_error_mean}")

    return recursive_tree


def run_non_ml_level_model_crossvalidation(
        ml_experiment_name: str, experiment_directory: str,
        device_name: str, feature_sets: List[str],
        training_strategy: str, standardize_features: bool,
        polynomial_features: bool, normalize_loss_scale: bool,
        tanh_scalar: float, num_epochs: int, log_train_diff: bool,
        save_nodes: bool
    ) -> Dict[str, float]:
    """
    Runs ML-level model cross-validation (leave-one-out) models
    and returns results dictionary.
    """

    feature_list = get_feature_types_to_feature_list(feature_sets, False, False)
    dataset_directory = os.path.join("datasets", device_name)

    serialization_directory = "serialization_directory"
    ground_feature_filepaths = [os.path.join(
        serialization_directory, ml_experiment_name, "ml_level_predictions.csv"
    )]
    module_feature_filepaths = [
        os.path.join(dataset_directory, f"module_level_features.csv"),
    ]
    model_feature_filepaths = [
        os.path.join(dataset_directory, f"model_level_features.csv"),
    ]

    # leaf_node_energies_true = get_leaf_node_energies(ground_feature_filepaths, model_list, "energy_mean")
    leaf_node_energies_predicted = get_leaf_node_energies(ground_feature_filepaths, MODEL_LIST, "predicted_energy")

    all_feature_filepaths = (
        ground_feature_filepaths
        +module_feature_filepaths
        +model_feature_filepaths
    )
    add_num_parameters = True
    add_input_size = True

    # Set these to False, to be able to serialize tree jsons
    # standardize_features = False
    # polynomial_features = False

    node_features, node_energies = get_node_features_and_energies(all_feature_filepaths,
                                                                  MODEL_LIST, feature_list,
                                                                  add_num_parameters,
                                                                  add_input_size,
                                                                  standardize_features,
                                                                  polynomial_features)
    num_features = list(node_features.values())[0].shape[-1]

    module_rows = feature_filepaths_to_rows(module_feature_filepaths)
    model_rows = feature_filepaths_to_rows(model_feature_filepaths)

    results_dict = {}

    model_level_metric = Average()
    module_level_metric = Average()

    for test_model_name in MODEL_LIST:
        print(f"\nTest  on {test_model_name}")

        results_dict[test_model_name] = {}

        train_model_list = [model_name for model_name in MODEL_LIST
                            if model_name != test_model_name]
        print(f"Train on {train_model_list}")

        if training_strategy == "unstructured":
            setting = "free"
        elif training_strategy == "none":
            setting = "exact_one"
        else:
            setting = "close_to_one"

        recursive_tree = RecursiveTreeComputation(
            num_features=num_features,weighting_setting=setting,
            tanh_scalar=tanh_scalar, training_strategy=training_strategy,
            normalize_loss_scale=normalize_loss_scale
        )

        if log_train_diff:
            # Evaluation (BEFORE)
            print("Result before:")
            results = predict_and_evaluate(recursive_tree, [test_model_name], model_rows,
                                           num_features, node_energies,
                                           leaf_node_energies_predicted, node_features, True, False)
            results.pop("predictions")
            print("Model level scores:")
            print(json.dumps(results, indent=4))
            results = predict_and_evaluate(recursive_tree, [test_model_name], module_rows,
                                           num_features, node_energies,
                                           leaf_node_energies_predicted, node_features, False, False)
            results.pop("predictions")
            print("Module level scores:")
            print(json.dumps(results, indent=4))

        if training_strategy != "none":
            recursive_tree = train_weighting_model(recursive_tree, train_model_list, model_rows,
                                                   num_features, node_energies,
                                                   leaf_node_energies_predicted, node_features,
                                                   num_epochs)

        if log_train_diff:
            # Evaluation (AFTER)
            print("Result after:")
        save_annotated_trees_dir = os.path.join(experiment_directory, "trees") if save_nodes else None
        results = predict_and_evaluate(recursive_tree, [test_model_name], model_rows,
                                       num_features, node_energies,
                                       leaf_node_energies_predicted, node_features, True,
                                       save_annotated_trees_dir)
        print("Model level scores:")
        predictions = results.pop("predictions")
        model_level_metric(len(predictions)*results["root_percentage_error"], len(predictions))
        results_dict[test_model_name]["model_level"] = results
        print(json.dumps(results, indent=4))

        if not save_nodes:
        # Module level prediction takes time and don't need annotated trees.
            results = predict_and_evaluate(recursive_tree, [test_model_name], module_rows,
                                           num_features, node_energies,
                                           leaf_node_energies_predicted, node_features, False,
                                           False)
            print("Module level scores:")
            predictions = results.pop("predictions")
            module_level_metric(len(predictions)*results["root_percentage_error"], len(predictions))
            results_dict[test_model_name]["module_level"] = results
            print(json.dumps(results, indent=4))

    results_dict["overall"] = {
        "model_level": round(model_level_metric.get_value(), 1),
    }
    if not save_nodes:
        results_dict["overall"]["module_level"] = round(module_level_metric.get_value(), 1)

    return results_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", type=str, help="experiment name: "
                        "experiment_configs/<experiment_name>.json should be config file path. "
                        "If you pass __print_all__ here, it'll print all the commands you have to run "
                        "to run to run cross-validation for the configs used in the paper. And "
                        "__exec_all__ will run them all.")
    args = parser.parse_args()

    if args.experiment_name in ("__print_all__", "__exec_all__"):
        all_ml_level_configs = generate_all_experiment_configs()["non_ml_level"]
        exec_commands = [
            f"nohup python non_ml_level_crossvalidation.py {config['non_ml_experiment_name']} "
            f"> nohup_logs/{config['non_ml_experiment_name']}.log &"
            for config in all_ml_level_configs
        ]

    if args.experiment_name == "__print_all__":
        print("Run the following commands. Running in background with nonhup is optional.")
        print("\n".join(exec_commands))
        exit()

    if args.experiment_name == "__exec_all__":
        print("Running the following commands:")
        for exec_command in exec_commands:
            print(exec_command)
            subprocess.call(exec_command, shell=True)
        exit()

    config = parse_experiment_config(args.experiment_name)
    assert config["non_ml_experiment_name"] == args.experiment_name, \
        "Mismatching experiment names in the config and the one passed as argument."
    ml_experiment_name = config["ml_experiment_name"]

    print("Running Non-ML-level cross-validation with following config:")
    print(json.dumps(config, indent=4))

    serialization_directory = "serialization_directory"
    experiment_directory = os.path.join(serialization_directory, args.experiment_name)
    os.makedirs(experiment_directory, exist_ok=True)

    used_config_path = os.path.join(experiment_directory, "non_ml_level_config.json")
    results_filepath = os.path.join(experiment_directory, "non_ml_level_results.json")

    write_json(config, used_config_path)
    results_dict = run_non_ml_level_model_crossvalidation(
        ml_experiment_name=ml_experiment_name, experiment_directory=experiment_directory,
        device_name=config["device_name"], feature_sets=config["feature_sets"],
        training_strategy=config["training_strategy"], polynomial_features=config["polynomial_features"],
        standardize_features=config["standardize_features"],
        normalize_loss_scale=config["normalize_loss_scale"], tanh_scalar=config["tanh_scalar"],
        num_epochs=config["num_epochs"], log_train_diff=config["log_train_diff"], save_nodes=config["save_nodes"]
    )
    write_json(results_dict, results_filepath)


if __name__ == '__main__':
    main()
