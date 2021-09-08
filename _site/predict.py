"""End2End Predict"""
import argparse

from lib.utils import read_jsonl, write_jsonl
from lib.end2end import end2end_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predict_data_filepath",
        type=str,
        help="filepath to data to predict on (jsonl format).",
    )
    parser.add_argument(
        "output_data_filepath",
        type=str,
        help="filepath to save predictions in (jsonl format).",
    )
    parser.add_argument(
        "model_directory", type=str, help="directory to save the models in."
    )
    args = parser.parse_args()

    predict_tree_jsons = read_jsonl(args.predict_data_filepath)
    predict_tree_jsons = end2end_predict(args.model_directory, predict_tree_jsons)
    write_jsonl(predict_tree_jsons, args.output_data_filepath)


if __name__ == "__main__":
    main()
