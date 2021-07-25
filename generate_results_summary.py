from typing import List, Dict

import pandas as pd

from generate_all_experiment_configs import generate_all_experiment_configs


def generate_non_ml_level_results_dataframe(configs: List[Dict]):
    """
    Generate summary results dataframe for non-ml level runs.
    """

    df_column_keys = [
        "dataset_name",
        "normalize_loss_scale",
        "polynomial_features",
        "feature_sets",
        "training_strategy",
    ]

    # TODO(cleanup): Change this?
    model_list = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "roberta-base",
        "distilgpt2",
        "openai-gpt",
        "gpt2",
    ]

    df_dict = defaultdict(list)
    for config in configs:

        # TODO(cleanup): What was this for?
        if config["save_nodes"]:
            continue

        experiment_name = experiment_config_to_name(config)
        used_config_path = f"serialization_directory/{experiment_name}/config.json"
        results_path = f"serialization_directory/{experiment_name}/results.json"

        if not os.path.exists(results_path):
            continue

        with open(used_config_path, "r") as file:
            used_config = json.load(file)

        for key in df_column_keys:
            df_dict[key].append(used_config[key])

        with open(results_path, "r") as file:
            results = json.load(file)

        df_dict["overall_model_level"].append(results["overall"]["model_level"])
        df_dict["overall_module_level"].append(results["overall"]["module_level"])

        for model_name in model_list:
            df_dict[f"{model_name}_model_level"].append(
                results[model_name]['model_level']['root_percentage_error']
            )
            df_dict[f"{model_name}_module_level"].append(
                results[model_name]['module_level']['root_percentage_error']
            )

        # assert all keys lists are of the same size.
        assert len(set([len(_list) for _list in df_dict.values()])) == 1

    dataframe = pd.DataFrame.from_dict(df_dict)
    return dataframe


def generate_ml_level_results_dataframe(configs: List[Dict]):
    """
    Generate summary results dataframe for ml level runs.
    """

    df_column_keys = [
        "device_name",
        "feature_sets"
    ]

    df_dict = defaultdict(list)
    for config in configs:

        experiment_name = experiment_config_to_name(config)
        used_config_path = f"serialization_directory/{experiment_name}/config.json"
        results_path = f"serialization_directory/{experiment_name}/ml_level_results.json"

        with open(used_config_path, "r") as file:
            used_config = json.load(file)

        assert used_config["device_name"] == config["device_name"]
        assert used_config["feature_sets"] == config["feature_sets"]

        df_dict["device_name"].append(config["device_name"])
        df_dict["feature_sets"].append(config["feature_sets"])

        results_path = f"serialization_directory/{experiment_name}/ml_level_results.json"
        with open(results_path, "r") as file:
            results_dict = json.load(file)

        for key, value in results_dict.items():
            df_dict[key].append(value)

    dataframe = pd.DataFrame.from_dict(df_dict)
    return dataframe


def main():

    all_experiment_configs = generate_all_experiment_configs()
    ml_level_configs = all_experiment_configs["ml_level"]
    non_ml_level_configs = all_experiment_configs["non_ml_level"]

    results_path = "results/ml_results.csv"
    dataframe = generate_ml_level_results_dataframe(ml_level_configs)
    dataframe.to_csv(results_path)
    print(f"Saved ml-level results in {results_path}")

    results_path = "results/model_module_results.csv"
    dataframe = generate_non_ml_level_results_dataframe(non_ml_level_configs)
    dataframe.to_csv("results/model_module_results.csv")
    print(f"Saved non-ml-level results in {results_path}")

if __name__ == '__main__':
    main()
