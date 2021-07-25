import argparse
import json
from typing import List, Dict

from lib.tree_node import TreeNode
from lib.utils import read_jsonl, get_percentage_error_list
from lib.end2end import evaluate_predictions_from_filepaths


def main():
    parser = argparse.ArgumentParser("Evaluates predictions and prints results")
    parser.add_argument(
        "ground_truth_filepath",
        type=str,
        help="path to trees containing ground_truths (jsonl format).",
    )
    parser.add_argument(
        "prediction_filepath",
        type=str,
        help="path to trees containing predictions (jsonl format).",
    )
    parser.add_argument(
        "--node_types",
        type=str,
        default="ml,module,model",
        help="comma seperated node_types. node_types are ml, module, model.",
    )
    args = parser.parse_args()

    results = evaluate_predictions_from_filepaths(
        args.ground_truth_filepath, args.prediction_filepath, args.node_types.split(",")
    )

    print("Percentage Error Results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
