"""
Generates all the experiment configs used in the paper.
"""
from typing import List, Dict
import argparse
import itertools
import os

from utils import write_json


def experiment_config_to_name(config: Dict):
    to_str = lambda item: "-".join(item) if isinstance(item, list) else item
    experiment_name = "___".join([f"{key}__{to_str(config[key])}"
                                  for key in sorted(config.keys())]).lower()
    return experiment_name


def generate_all_experiment_configs() -> Dict[str, List[Dict]]:
    """
    Generates all the experiment configs used in the paper.
    """

    device_names = ["device_1", "device_2"] # name change for consistency with paper (qpc->1, jpc-> 2)
    normalize_loss_scales = [True]
    list_polynomial_features = [True]

    list_feature_sets = [
        ["model_specs"],
        ["resource_utilization"],
        ["model_specs", "resource_utilization"]
    ]

    training_strategies = [
        "unstructured", # name change for consistency with paper (isolated -> unstructured)
        "piecewise", # name change for consistency with paper (piecewise -> stepwise)
        "end2end",
    ]

    list_save_nodes = [
        True
    ]

    configurations = itertools.product(
        device_names,
        normalize_loss_scales,
        list_polynomial_features,
        list_feature_sets,
        training_strategies,
        list_save_nodes
    )

    ml_level_configs = []
    non_ml_level_configs = []
    for configuration in configurations:

        (device_name, normalize_loss_scale, polynomial_features,
         feature_sets, training_strategy, save_nodes) = configuration

        ml_level_config = {
            "device_name": device_name,
            "feature_sets": feature_sets
        }

        ml_experiment_name = experiment_config_to_name(ml_level_config)
        ml_level_config["ml_experiment_name"] = ml_experiment_name

        non_ml_level_config = {
            "device_name": device_name,
            "normalize_loss_scale": normalize_loss_scale,
            "polynomial_features": polynomial_features,
            "feature_sets": feature_sets,
            "training_strategy": training_strategy,
            "save_nodes": save_nodes
        }
        non_ml_experiment_name = experiment_config_to_name(non_ml_level_config)
        non_ml_level_config["ml_experiment_name"] = ml_experiment_name
        non_ml_level_config["non_ml_experiment_name"] = non_ml_experiment_name

        if ml_level_config not in ml_level_configs:
            ml_level_configs.append(ml_level_config)
        non_ml_level_configs.append(non_ml_level_config)

    return {
        "ml_level": ml_level_configs,
        "non_ml_level": non_ml_level_configs
    }

def main():

    all_experiment_configs = generate_all_experiment_configs()
    ml_level_experiment_configs = all_experiment_configs["ml_level"]
    non_ml_level_experiment_configs = all_experiment_configs["non_ml_level"]

    print("\nWriting ML-level experiment configs:")
    for config in ml_level_experiment_configs:
        experiment_name = config["ml_experiment_name"]
        experiment_path = os.path.join("experiment_configs", experiment_name + ".json")
        write_json(config, experiment_path)

    print("\nWriting Non-ML-level experiment configs:")
    for config in non_ml_level_experiment_configs:
        experiment_name = config["non_ml_experiment_name"]
        experiment_path = os.path.join("experiment_configs", experiment_name + ".json")
        write_json(config, experiment_path)


if __name__ == '__main__':
    main()
