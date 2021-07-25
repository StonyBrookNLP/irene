import argparse
from typing import Tuple, List, Dict
import json
import copy
import pickle  # may be change to dill?
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from lib.tree_node import TreeNode

np.random.seed(13370)


def train_linear_regressor(features: np.array, ground_truths: np.array) -> Tuple:
    """
    Scales data and trains a simple linear regressor.
    """

    regressor = linear_model.LinearRegression()
    scale_standardizer = StandardScaler().fit(features)
    features = scale_standardizer.transform(features)
    transformations = [scale_standardizer]

    regressor = regressor.fit(features, ground_truths)
    return regressor, transformations


def predict_linear_regressor(
    regressor, transformations: List, features: np.array, ids: List[str]
) -> Dict[str, float]:
    """
    Generates prediction using trained regressor on the passed features
    and returns a dictionary of id to predictions.
    """

    for transformation in transformations:
        features = transformation.transform(features)

    predicted_outputs = regressor.predict(features)
    id_to_predicted_values = {
        id_: pred for id_, pred in zip(list(ids), list(predicted_outputs))
    }

    return id_to_predicted_values


def train_ml_level_models(train_trees: List[TreeNode]) -> Tuple[Dict, Dict]:
    """
    Trains ML-level regressor on the leaf nodes of training trees and outputs
    trained regressor and scalars.
    """

    operationwise_ml_level_instances = defaultdict(list)
    for tree in train_trees:
        for operation_type, ml_level_instances in tree.get_ml_level_data().items():
            operationwise_ml_level_instances[operation_type].extend(ml_level_instances)

    operationwise_ml_level_model = {}
    operationwise_ml_level_transformations = {}
    for operation_type, ml_level_instances in operationwise_ml_level_instances.items():
        features = np.stack(
            [np.array(instance["features"]) for instance in ml_level_instances], axis=0
        )
        ground_truths = np.array(
            [instance["gold_energy"] for instance in ml_level_instances]
        )
        regressor, transformations = train_linear_regressor(
            features=features, ground_truths=ground_truths
        )
        operationwise_ml_level_model[operation_type] = regressor
        operationwise_ml_level_transformations[operation_type] = transformations

    return operationwise_ml_level_model, operationwise_ml_level_transformations


def predict_ml_level_models(
    operationwise_ml_level_model: Dict,
    operationwise_ml_level_transformations: Dict,
    predict_trees: List[TreeNode],
) -> List[TreeNode]:
    """
    Runs regressor on the leaf/ml-level nodes of the predic_trees and saves
    the predicted_energy field into it. Returns predicted_energy annotated trees.
    """
    assert set(operationwise_ml_level_model.keys()) == set(
        operationwise_ml_level_transformations.keys()
    )

    predict_trees = copy.deepcopy(predict_trees)
    for predict_tree in predict_trees:
        operationwise_ml_level_instances = predict_tree.get_ml_level_data()

        for (
            operation_type,
            ml_level_instances,
        ) in operationwise_ml_level_instances.items():

            if operation_type not in operationwise_ml_level_model:
                raise Exception(
                    f"Given model isn't trained on operation_type {operation_type}"
                )

            regressor = operationwise_ml_level_model[operation_type]
            transformations = operationwise_ml_level_transformations[operation_type]

            features = np.stack(
                [np.array(instance["features"]) for instance in ml_level_instances],
                axis=0,
            )
            ids = [instance["id"] for instance in ml_level_instances]
            id_to_predicted_values = predict_linear_regressor(
                regressor, transformations, features, ids
            )
            predict_tree.update_tree_node_attributes(
                "predicted_energy", id_to_predicted_values
            )

    return predict_trees
