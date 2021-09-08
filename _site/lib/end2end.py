import os
import json
from copy import deepcopy
from typing import List, Dict
from collections import defaultdict

import pandas as pd

from lib.tree_node import TreeNode
from lib.ml_level import train_ml_level_models, predict_ml_level_models
from lib.non_ml_level import train_non_ml_level_model, predict_non_ml_level_model
from lib.utils import (
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
    load_ml_level_model_transformations,
    load_non_ml_level_model_transformations,
    save_ml_level_model_transformations,
    save_non_ml_level_model_transformations,
    get_feature_types_to_feature_list,
    get_percentage_error_list,
    IRENE_CONFIG,
)


def end2end_train(
    train_tree_jsons: List[Dict], model_directory: str, config: Dict = None
):
    """End2End Train"""
    config = deepcopy(config or IRENE_CONFIG)
    used_config = deepcopy(config)

    feature_list = get_feature_types_to_feature_list(config.pop("feature_sets"))
    train_trees = TreeNode.read_from_jsons(train_tree_jsons, feature_list)

    os.makedirs(model_directory, exist_ok=True)

    (
        operationwise_ml_level_model,
        operationwise_ml_level_transformations,
    ) = train_ml_level_models(train_trees)
    save_ml_level_model_transformations(
        operationwise_ml_level_model,
        operationwise_ml_level_transformations,
        model_directory,
    )

    train_trees = predict_ml_level_models(
        operationwise_ml_level_model,
        operationwise_ml_level_transformations,
        train_trees,
    )

    non_ml_level_model, non_ml_transformations = train_non_ml_level_model(
        train_trees, **config
    )
    save_non_ml_level_model_transformations(
        non_ml_level_model, non_ml_transformations, model_directory
    )

    used_config_filepath = os.path.join(model_directory, "config.json")
    write_json(used_config, used_config_filepath)


def end2end_predict(model_directory: str, predict_tree_jsons: List[Dict]) -> List[Dict]:
    """End2End Predict"""

    used_config_filepath = os.path.join(model_directory, "config.json")
    config = read_json(used_config_filepath)

    feature_list = get_feature_types_to_feature_list(config.pop("feature_sets"))
    predict_trees = TreeNode.read_from_jsons(predict_tree_jsons, feature_list)

    (
        operationwise_ml_level_model,
        operationwise_ml_level_transformations,
    ) = load_ml_level_model_transformations(model_directory)
    predict_trees = predict_ml_level_models(
        operationwise_ml_level_model,
        operationwise_ml_level_transformations,
        predict_trees,
    )

    (
        non_ml_level_model,
        non_ml_level_transformations,
    ) = load_non_ml_level_model_transformations(model_directory)
    predict_trees = predict_non_ml_level_model(
        non_ml_level_model, non_ml_level_transformations, predict_trees
    )
    predict_tree_jsons = TreeNode.write_to_jsons(predict_trees)
    return predict_tree_jsons


def evaluate_predictions_from_jsons(  # TODO: Change name to end2end_evaluate ?
    ground_truth_tree_jsons: List[Dict],
    prediction_tree_jsons: List[Dict],
    node_types: List[str] = None,
) -> Dict[str, float]:
    """
    Evaluates predictions for given node_types (ml, module, model)
    from the passed ground_truth and predictions trees.
    """
    node_types = node_types or ["ml", "module", "model"]

    ground_truth_trees = TreeNode.read_from_jsons(ground_truth_tree_jsons, [])
    predictions_trees = TreeNode.read_from_jsons(prediction_tree_jsons, [])

    node_type_to_percentage_errors = {}
    for node_type in node_types:
        assert node_type in ("model", "module", "ml")

        id_to_gold_energy = {}
        for tree in ground_truth_trees:
            for attribute_object in tree.get_subtree_nodes_attributes(
                [node_type], ["id", "gold_energy"]
            ):
                id_to_gold_energy[attribute_object["id"]] = attribute_object[
                    "gold_energy"
                ]

        id_to_predicted_energy = {}
        for tree in predictions_trees:
            for attribute_object in tree.get_subtree_nodes_attributes(
                [node_type], ["id", "predicted_energy"]
            ):
                id_to_predicted_energy[attribute_object["id"]] = attribute_object[
                    "predicted_energy"
                ]

        expected_ids = id_to_gold_energy.keys()
        gold_energies = [id_to_gold_energy[id_] for id_ in expected_ids]
        predicted_energies = []
        for id_ in expected_ids:
            predicted_energy = id_to_predicted_energy.get(id_, None)

            if not predicted_energy:
                print(
                    f"WARNING: No predicted energy found for node-id {id_}. Force setting 0."
                )
                predicted_energy = 0

            predicted_energies.append(predicted_energy)

        percentage_error = get_percentage_error_list(gold_energies, predicted_energies)
        node_type_to_percentage_errors[node_type] = round(percentage_error, 2)

    return node_type_to_percentage_errors


def evaluate_predictions_from_filepaths(
    ground_truth_filepath: str, prediction_filepath: str, node_types: List[str]
) -> Dict[str, float]:
    """
    Evaluates predictions for given node_types (ml, module, model)
    from the passed ground_truth and predictions filepath.
    """

    ground_truth_tree_jsons = read_jsonl(ground_truth_filepath)
    prediction_tree_jsons = read_jsonl(prediction_filepath)

    return evaluate_predictions_from_jsons(
        ground_truth_tree_jsons, prediction_tree_jsons, node_types
    )


def end2end_crossvalidate(
    tree_jsons: List[Dict], model_directory: str, config: Dict = None
):
    """End2End CrossValidation (Leave-one-out model)"""
    config = deepcopy(config or IRENE_CONFIG)

    model_to_tree_jsons = defaultdict(list)
    for tree_json in tree_jsons:
        model_name = tree_json["model_name"]
        model_to_tree_jsons[model_name].append(tree_json)

    model_names = list(model_to_tree_jsons.keys())

    report_pd_dict = defaultdict(list)
    for test_model_name in model_names:
        train_model_names = [
            model_name for model_name in model_names if model_name != test_model_name
        ]

        print(f"\nTesting  on {test_model_name}")
        print(f"Training on {', '.join(train_model_names)}")

        train_tree_jsons = [
            deepcopy(tree_json)
            for model_name in train_model_names
            for tree_json in model_to_tree_jsons[model_name]
        ]

        predict_tree_jsons = deepcopy(model_to_tree_jsons[test_model_name])

        directory = os.path.join(
            model_directory, "crossvalidation", f"leave_{test_model_name}"
        )
        os.makedirs(directory, exist_ok=True)

        end2end_train(train_tree_jsons, directory, config)
        predict_tree_jsons = end2end_predict(directory, predict_tree_jsons)

        prediction_filepath = os.path.join(directory, "predictions.jsonl")
        write_jsonl(predict_tree_jsons, prediction_filepath)

        percentage_errors = evaluate_predictions_from_jsons(
            predict_tree_jsons, predict_tree_jsons
        )
        print("percentage errors:")
        print(json.dumps(percentage_errors, indent=4))

        results_filepath = os.path.join(directory, "percentage_errors.json")
        write_json(percentage_errors, results_filepath)

        report_pd_dict[f"left-model-name"].append(test_model_name)
        report_pd_dict[f"ml % error"].append(percentage_errors["ml"])
        report_pd_dict[f"module % error"].append(percentage_errors["module"])
        report_pd_dict[f"model % error"].append(percentage_errors["model"])

    average = lambda items: round(sum(items) / len(items), 3) if items else 0.0
    report_pd_dict[f"left-model-name"].append("overall")
    report_pd_dict[f"ml % error"].append(average(report_pd_dict[f"ml % error"]))
    report_pd_dict[f"module % error"].append(average(report_pd_dict[f"module % error"]))
    report_pd_dict[f"model % error"].append(average(report_pd_dict[f"model % error"]))

    report_df = pd.DataFrame.from_dict(report_pd_dict).round(2)
    print("Percentage Error - Cross Validation Report")
    print(report_df)
