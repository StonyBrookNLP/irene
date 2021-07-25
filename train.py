"""End2End Train"""
import argparse

from lib.utils import read_json, read_jsonl
from lib.end2end import end2end_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_filepath",
        type=str,
        help="filepath to the training data (jsonl format).",
    )
    parser.add_argument(
        "model_directory", type=str, help="directory to save the models in."
    )
    parser.add_argument("--config_filepath", type=str, help="filepath to config.json.")
    args = parser.parse_args()

    config = read_json(args.config_filepath) if args.config_filepath else None
    train_tree_jsons = read_jsonl(args.train_data_filepath)
    end2end_train(train_tree_jsons, args.model_directory, config)


if __name__ == "__main__":
    main()
