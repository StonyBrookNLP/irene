"""
Train and evaluate cross-validation (leave-one-out) models on
ml-level (leaf) nodes data.
"""
import uuid
import os
import math
import csv
from collections import defaultdict
from functools import lru_cache
from typing import Union, List, Dict, Any, Tuple
import argparse
import json
import copy

from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np
import pandas as pd

from utils import (
    write_json, parse_experiment_config, get_model, get_num_parameters, get_percentage_error_list,
    get_feature_types_to_feature_list, MODEL_LIST, ML_OPERATIONS_LIST
)
from generate_all_experiment_configs import generate_all_experiment_configs


def get_rows_to_ids_features_outputs(
        rows: List[Dict],
        feature_list: List[str],
        output_key: str,
        add_num_parameters: bool = True,
        add_input_size: bool = True
    ) -> Dict[str, Any]:
    """
    Take a list of rows from dataset input csv, and outputs
    ids, features, gold_outputs, which can be gold-energy or something
    else depending on the passed `output_key`.
    """

    list_features = []
    list_outputs = []
    list_ids = []

    for row in rows:

        id_ = row["id"]
        features = [float(row[key]) for key in feature_list]

        if add_input_size:
            input_size = float(row["batch_size"])*float(row["seq_len"])
            features.insert(0, input_size)

        if add_num_parameters:
            model_name = row['model_name']
            level_name = row['level_name']
            num_parameters = get_num_parameters(model_name, level_name)
            features.insert(0, num_parameters)

        output = float(row[output_key])

        list_features.append(features)
        list_outputs.append(output)
        list_ids.append(id_)

    ids = np.array(list_ids)
    features = np.array(list_features)
    outputs = np.array(list_outputs)

    ids_features_outputs = {
        "ids": ids,
        "features": features,
        "outputs": outputs
    }

    return ids_features_outputs


def read_ml_level_rows(
        feature_filepaths: List[str],
        model_list: List[str],
        ml_operations_list: List[str]
    ) -> Dict:
    """
    Reads ml-level data csv file and organizes the rows into model_name, operation_type.
    """

    nested_rows_dict = defaultdict(lambda: defaultdict(list))

    for feature_filepath in feature_filepaths:
        with open(feature_filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                model_name = row['model_name']
                level_name = row['level_name']
                operation_type = level_name.split(":")[1]

                if model_name not in model_list:
                    continue

                if operation_type not in ml_operations_list:
                    continue

                id_ = uuid.uuid4()
                row["id"] = id_

                nested_rows_dict[model_name][operation_type].append(row)

    return nested_rows_dict


def combine_ids_features_outputs(list_ids_features_outputs: List[Dict]) -> Dict[str, Any]:
    """
    Combines list input data (ids, features, gold_output) into one.
    """

    list_features = [ids_features_outputs["features"]
                     for ids_features_outputs in list_ids_features_outputs]

    list_outputs = [ids_features_outputs["outputs"]
                    for ids_features_outputs in list_ids_features_outputs]

    list_ids = [ids_features_outputs["ids"]
                for ids_features_outputs in list_ids_features_outputs]

    combined_list_ids_features_outputs = {
        "features": np.concatenate(list_features, axis=0),
        "outputs": np.concatenate(list_outputs, axis=0),
        "ids": [id_ for ids in list_ids for id_ in ids]
    }
    return combined_list_ids_features_outputs


def train_regressor(features: np.array,
                    gold_output: np.array,
                    ids: List[str],
                    regressor_type: str = "poly_linear",
                    standardize_scale: bool = True) -> Tuple:
    """
    Trains a regressor on (optionally standardized/scaled) input data.
    """

    transformations = []

    if regressor_type == "decision_tree":
        regressor = DecisionTreeRegressor(random_state=5)

    if regressor_type == "poly_linear":
        poly_reg = PolynomialFeatures(degree=3)
        features = poly_reg.fit_transform(features)
        transformations.append(poly_reg)
        regressor = linear_model.LinearRegression()

    if regressor_type == "linear":
        regressor = linear_model.LinearRegression()

    if standardize_scale:
        scale_standardizer = StandardScaler().fit(features)
        features = scale_standardizer.transform(features)
        transformations.append(scale_standardizer)

    regressor = regressor.fit(features, gold_output)
    return regressor, transformations


def regressor_predict_evaluate(regressor,
                               features: np.array,
                               gold_output: np.array,
                               ids: List[str],
                               transformations: List) -> Dict[str, float]:
    """
    Generates prediction using trained regressor on the passed
    features, and returns results dictionary.
    """

    for transformation in transformations:
        features = transformation.transform(features)

    predicted_outputs = regressor.predict(features)
    predictions = [{"id": id_, "true_ys": real, "pred_ys": pred}
                    for id_, real, pred in
                    zip(list(ids), list(gold_output), list(predicted_outputs))]

    true_ys = [e["true_ys"] for e in predictions]
    pred_ys = [e["pred_ys"] for e in predictions]

    precentage_error = get_percentage_error_list(true_ys, pred_ys)
    min_max_range = max(true_ys) - min(true_ys)

    result = {
        "precentage_error": precentage_error,
        "range": min_max_range,
        "predictions": predictions
    }

    return result

def run_ml_level_model_crossvalidation(
        device_name: str, feature_sets: List[str], predictions_filepath: str
    ) -> Dict[str, float]:
    """
    Runs ML-level model cross-validation (leave-one-out) models
    and returns results dictionary.
    """

    feature_list = get_feature_types_to_feature_list(feature_sets, False, False)
    output_key = "energy_mean"

    dataset_directory = os.path.join("datasets", device_name)
    feature_filepaths = [
        os.path.join(dataset_directory, f"ml1_level_features.csv"),
        os.path.join(dataset_directory, f"ml2_level_features.csv")
    ]

    ml_level_rows = read_ml_level_rows(feature_filepaths, MODEL_LIST, ML_OPERATIONS_LIST)

    id2row = {}
    ml_level_data = defaultdict(dict)
    for model_name, data in ml_level_rows.items():
        # print(model_name)
        for operation_type, _ in data.items():
            # print("\t" + operation_type)

            rows = ml_level_rows[model_name][operation_type]
            ids_features_outputs = get_rows_to_ids_features_outputs(
                rows, feature_list, output_key, True, True
            )
            ml_level_data[model_name][operation_type] = ids_features_outputs

            for row in rows:
                id2row[row["id"]] = row

    all_predictions = []
    modelwise_predictions = defaultdict(list)
    operationwise_predictions = defaultdict(list)
    for test_model_name, data in ml_level_rows.items():
        print("\nTest on "+ test_model_name)

        for operation_type, _ in data.items():
            print("")
            print("\t" + operation_type)

            if operation_type not in data:
                continue

            test_ids_features_outputs = ml_level_data[test_model_name][operation_type]

            train_model_names = []
            list_train_ids_features_outputs = []
            for train_model_name in MODEL_LIST:

                if train_model_name == test_model_name:
                    continue

                if operation_type not in ml_level_data[train_model_name]:
                    continue

                train_ids_features_outputs = ml_level_data[train_model_name][operation_type]
                list_train_ids_features_outputs.append(train_ids_features_outputs)
                train_model_names.append(train_model_name)

            print("\t" + "Train on: " + str(train_model_names))

            train_ids_features_outputs = combine_ids_features_outputs(list_train_ids_features_outputs)
            regressor, transformations = train_regressor(features=train_ids_features_outputs["features"],
                                                         gold_output=train_ids_features_outputs["outputs"],
                                                         ids=train_ids_features_outputs["ids"],
                                                         regressor_type="linear",
                                                         standardize_scale=True)

            test_ids_features_outputs = combine_ids_features_outputs([test_ids_features_outputs])
            result = regressor_predict_evaluate(regressor=regressor,
                                                features=test_ids_features_outputs["features"],
                                                gold_output=test_ids_features_outputs["outputs"],
                                                ids=test_ids_features_outputs["ids"],
                                                transformations=transformations
                                                )
            print("\t" + f"PERC error: {result['precentage_error']}")

            for prediction in result["predictions"]:
                id_ = prediction["id"]
                predicted_energy = prediction["pred_ys"]
                id2row[id_]["predicted_energy"] = predicted_energy

            all_predictions.extend(result["predictions"])
            modelwise_predictions[test_model_name].extend(result["predictions"])
            operationwise_predictions[operation_type].extend(result["predictions"])

    print("")
    results_dict = {}
    for operation_type, predictions in operationwise_predictions.items():
        true_ys = [e["true_ys"] for e in predictions]
        pred_ys = [e["pred_ys"] for e in predictions]
        model_precentage_error = get_percentage_error_list(true_ys, pred_ys)
        print(f"Operation {operation_type}: Percentage Error: {model_precentage_error}")
        results_dict[operation_type] = model_precentage_error


    print("")
    for model_name, predictions in modelwise_predictions.items():
        true_ys = [e["true_ys"] for e in predictions]
        pred_ys = [e["pred_ys"] for e in predictions]
        model_precentage_error = get_percentage_error_list(true_ys, pred_ys)
        print(f"Model {model_name}: Percentage Error: {model_precentage_error}")
        results_dict[model_name] = model_precentage_error

    all_true_ys = [e["true_ys"] for e in all_predictions]
    all_pred_ys = [e["pred_ys"] for e in all_predictions]
    overall_precentage_error = get_percentage_error_list(all_true_ys, all_pred_ys)
    print(f"\nOvrall PERC error: {overall_precentage_error}")
    results_dict["overall"] = overall_precentage_error

    header_columns = list(list(id2row.values())[0].keys())
    header_columns.remove("id")

    print(f"\nWriting in {predictions_filepath}")
    os.makedirs(os.path.dirname(predictions_filepath), exist_ok=True)
    with open(predictions_filepath, "w") as file:
        writer = csv.DictWriter(file, header_columns)
        writer.writeheader()

        for row in id2row.values():
            row.pop("id")
            assert "predicted_energy" in row
            writer.writerow(row)

    results_dict = {key: round(value, 2) for key, value in results_dict.items()}
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
        all_ml_level_configs = generate_all_experiment_configs()["ml_level"]
        exec_commands = [
            f"nohup python ml_level_crossvalidation.py {config['ml_experiment_name']} "
            f"> nohup_logs/{config['ml_experiment_name']}.log &"
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
    assert config["ml_experiment_name"] == args.experiment_name, \
        "Mismatching experiment names in the config and the one passed as argument."

    print("Running ML-level cross-validation with following config:")
    print(json.dumps(config, indent=4))

    serialization_directory = "serialization_directory"
    experiment_directory = os.path.join(serialization_directory, args.experiment_name)
    os.makedirs(experiment_directory, exist_ok=True)

    used_config_path = os.path.join(experiment_directory, "ml_level_config.json")
    predictions_filepath = os.path.join(experiment_directory, "ml_level_predictions.csv")
    results_filepath = os.path.join(experiment_directory, "ml_level_results.json")

    write_json(config, used_config_path)
    results_dict = run_ml_level_model_crossvalidation(
        config["device_name"], config["feature_sets"], predictions_filepath
    )
    write_json(results_dict, results_filepath)


if __name__ == '__main__':
    main()
