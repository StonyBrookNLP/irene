"""End2End CrossValidation (Leave-one-out model)"""
import argparse

from lib.end2end import end2end_crossvalidate
from lib.utils import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_filepath",
        type=str,
        help="filepath to to run crossvalidate on (jsonl format).",
    )
    parser.add_argument(
        "model_directory", type=str, help="directory to save the models in."
    )
    parser.add_argument("--config_filepath", type=str, help="filepath to config.json.")
    args = parser.parse_args()

    config = read_json(args.config_filepath) if args.config_filepath else None
    tree_jsons = read_jsonl(args.data_filepath)
    end2end_crossvalidate(tree_jsons, args.model_directory)


if __name__ == "__main__":
    main()
