import uuid
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import torch

from lib.utils import read_jsonl, write_jsonl, ML_OPERATIONS_LIST


@dataclass(unsafe_hash=True)
class TreeNode:

    id: str
    model_name: str
    batch_size: int
    seq_len: int
    node_type: str

    parent_name: str
    instance_type: str
    scope: str
    level: str
    level_name: str
    child_nodes: List["TreeNode"] = None

    feature_list: List[str] = None
    features: List[float] = None

    gold_energy: float = None  # Name change to ground_truth_energy
    predicted_energy: float = None

    loss: float = None
    subtree_loss: float = None
    subtree_error_sum: float = None
    subtree_error_count: int = None
    subtree_error_perc: float = None

    original_node_dict: Dict = None

    def description(self):
        print(
            f"NODE INFORMATION - Scope: {self.scope}, instance type: {self.instance_type}, "
            f"level: {self.level}, parent: {self.parent_name}"
        )

    @classmethod
    def read_from_json(
        cls,
        tree_dict: Dict,
        feature_list: List[str],
        model_name: str = None,
        seq_len: str = None,
        batch_size: str = None,
    ) -> "TreeNode":

        if "model_name" in tree_dict:
            tree_dict = deepcopy(tree_dict)
            root_is_model_node = True

            assert not any(
                (model_name, seq_len, batch_size)
            ), "Don't pass model_name, seq_len or batch_size at as arguments at root level."

            model_name = tree_dict.pop("model_name")
            seq_len = tree_dict.pop("seq_len")
            batch_size = tree_dict.pop("batch_size")
            tree_dict = tree_dict.pop("frontend_tree")

        else:
            root_is_model_node = False

        subtrees_dicts = tree_dict["child_nodes_obj"]

        features = []
        for feature_name in feature_list:

            if feature_name == "batch_size":
                feature_value = batch_size
            elif feature_name == "seq_len":
                feature_value = seq_len
            elif feature_name == "input_size":
                feature_value = batch_size * seq_len
            else:
                feature_value = tree_dict[feature_name]
            features.append(feature_value)

        node_type = tree_dict["type"]

        # Temporary fix for the format change, maybe update later?
        if node_type == "ml-np":
            node_type = "ml"

        if not subtrees_dicts:
            assert node_type == "ml", f"Leaf nodes must be ml typed. Found {node_type}"

        # There are some discrepancies in names: MatMul vs matmul.
        tree_dict["instance_type"] = tree_dict["instance_type"].lower()

        # This is just to make it compatible with old-code
        level_name = f"{tree_dict['scope']}:{tree_dict['instance_type']}"

        attributes_dict = {
            "id": tree_dict["id"],
            "scope": tree_dict["scope"],
            "level": tree_dict["level"],
            "parent_name": tree_dict["parent_name"],
            "instance_type": tree_dict["instance_type"],
            "level_name": level_name,
            "features": features,
            "model_name": model_name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "node_type": node_type,
            "gold_energy": tree_dict.get("ground_truth_energy", None),
            "predicted_energy": tree_dict.get("predicted_energy", None),
            "original_node_dict": tree_dict,  # For getting the predictions back in original format.
        }

        tree_node = TreeNode(**attributes_dict)
        tree_node.child_nodes = [
            cls.read_from_json(
                subtree_dict,
                feature_list=feature_list,
                model_name=model_name,
                seq_len=seq_len,
                batch_size=batch_size,
            )
            for subtree_dict in subtrees_dicts
        ]

        return tree_node

    def write_to_json(self, root_is_model_node: bool = True) -> Dict:

        child_nodes = self.child_nodes

        tree_node_dict = self.original_node_dict
        tree_node_dict["id"] = self.id
        tree_node_dict["predicted_energy"] = float(self.predicted_energy)

        tree_node_dict["child_nodes_obj"] = [
            child_node.write_to_json(False) for child_node in child_nodes
        ]
        if root_is_model_node:
            tree_node_dict = {
                "model_name": self.model_name,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "frontend_tree": tree_node_dict,
            }
        return deepcopy(tree_node_dict)

    @classmethod
    def read_from_jsons(
        cls, jsons: List[Dict], feature_list: List[str]
    ) -> List["TreeNode"]:
        return [TreeNode.read_from_json(tree_dict, feature_list) for tree_dict in jsons]

    @classmethod
    def write_to_jsons(cls, tree_nodes: List["TreeNode"]) -> List[Dict]:
        return [tree_node.write_to_json() for tree_node in tree_nodes]

    def get_subtree_nodes(self, node_types: List[str]) -> List["TreeNode"]:
        """
        Enumerates all ML-level (leaf) nodes of the tree.
        """
        for node_type in node_types:
            assert node_type in ("model", "module", "ml")

        if self.node_type in node_types:
            yield self

        for child in self.child_nodes:
            for leaf in child.get_subtree_nodes(node_types):
                yield leaf

    def update_tree_node_attributes(
        self, attribute: str, node_id_to_attribute_value: Dict[str, Any]
    ):
        """
        Iterates through the full tree and updates attribute values for the given attribute
        of the node with the matching node-id.
        """
        if self.id in node_id_to_attribute_value:
            setattr(self, attribute, node_id_to_attribute_value[self.id])

        for child_node in self.child_nodes:
            child_node.update_tree_node_attributes(
                attribute, node_id_to_attribute_value
            )

    def get_subtree_nodes_attributes(
        self, node_types: List[str], attributes: List[str]
    ) -> List[Dict]:
        """
        Enumerates nodes of the given node_types and extracts the given
        attributes of each of them into a list of dictionaries.
        """

        nodes_attributes = []
        for node in self.get_subtree_nodes(node_types):
            nodes_attributes.append(
                {attribute: getattr(node, attribute) for attribute in attributes}
            )

        return nodes_attributes

    def get_ml_level_data(self, output_attr: str = "gold_energy") -> Dict[str, List]:
        """
        Loads ML-level data (ids, features and outputs/energies) for each operation-type
        """
        nodes_attributes = self.get_subtree_nodes_attributes(
            ["ml"], ["id", "features", "gold_energy", "level_name", "instance_type"]
        )  # level_name was used for legacy code.

        for node_attributes in nodes_attributes:
            node_attributes["ids"] = node_attributes["id"]
            node_attributes["outputs"] = node_attributes[output_attr]

        operationwise_nodes_attributes = defaultdict(list)

        for node_attributes in nodes_attributes:
            # # level_name was used for legacy code:
            # level_name = node_attributes["level_name"]
            # operation_type = level_name.split(":")[1]

            operation_type = node_attributes["instance_type"]
            if operation_type in ML_OPERATIONS_LIST:
                operationwise_nodes_attributes[operation_type].append(node_attributes)
            else:
                print(
                    f"WARNING: No ML-level model found for operation_type: {operation_type}"
                )

        return operationwise_nodes_attributes

    @classmethod
    def prepare_for_non_ml_training(
        cls,
        tree_nodes: List["TreeNode"],
        standardize_features: bool = True,
        polynomial_features: bool = True,
    ) -> Tuple[List["TreeNode"], List]:
        """
        Runs feature transformations (standardizer and polynomial expander), save them as tensors
        and then also saves gold_energy as tensors.
        """

        nodes_attributes = []
        for tree_node in tree_nodes:
            nodes_attributes.extend(
                tree_node.get_subtree_nodes_attributes(
                    ["ml", "module", "model"], ["id", "features", "gold_energy"]
                )
            )

        assert len(
            set([node_attributes["id"] for node_attributes in nodes_attributes])
        ) == len(nodes_attributes), "Node-ids need to be unique, but they aren't."

        all_features = np.stack(
            [
                np.array(node_attributes["features"])
                for node_attributes in nodes_attributes
            ],
            axis=0,
        )

        transformations = []

        if standardize_features:
            scale_standardizer = StandardScaler().fit(all_features)
            all_features = scale_standardizer.transform(all_features)
            transformations.append(scale_standardizer)

        if polynomial_features:
            poly_reg = PolynomialFeatures(degree=3).fit(all_features)
            all_features = poly_reg.transform(all_features)
            transformations.append(poly_reg)

        all_features = torch.tensor(all_features, dtype=torch.float32)

        id_to_features = {
            node_attributes["id"]: features.unsqueeze(0)
            for node_attributes, features in zip(nodes_attributes, all_features)
        }
        for tree_node in tree_nodes:
            tree_node.update_tree_node_attributes("features", id_to_features)

        id_to_gold_energy = {
            node_attributes["id"]: torch.tensor(node_attributes["gold_energy"])
            for node_attributes in nodes_attributes
        }
        for tree_node in tree_nodes:
            tree_node.update_tree_node_attributes("gold_energy", id_to_gold_energy)

        return tree_nodes, transformations

    @classmethod
    def prepare_for_non_ml_predicting(
        cls,
        tree_nodes: List["TreeNode"],
        transformations: List,
    ) -> List["TreeNode"]:
        """
        Runs transformations trained on the training data, and changes
        features/gold_energy fields to tensors.
        """

        nodes_attributes = []
        for tree_node in tree_nodes:
            nodes_attributes.extend(
                tree_node.get_subtree_nodes_attributes(
                    ["ml", "module", "model"], ["id", "features", "gold_energy"]
                )
            )

        assert len(
            set([node_attributes["id"] for node_attributes in nodes_attributes])
        ) == len(nodes_attributes), "Node-ids need to be unique, but they aren't."

        all_features = np.stack(
            [
                np.array(node_attributes["features"])
                for node_attributes in nodes_attributes
            ],
            axis=0,
        )

        for transformation in transformations:
            all_features = transformation.transform(all_features)

        all_features = torch.tensor(all_features, dtype=torch.float32)

        id_to_features = {
            node_attributes["id"]: features.unsqueeze(0)
            for node_attributes, features in zip(nodes_attributes, all_features)
        }
        for tree_node in tree_nodes:
            tree_node.update_tree_node_attributes("features", id_to_features)

        id_to_gold_energy = {
            node_attributes["id"]: (
                torch.tensor(node_attributes["gold_energy"])
                if node_attributes.get("gold_energy", None)
                else None
            )
            for node_attributes in nodes_attributes
        }
        for tree_node in tree_nodes:
            tree_node.update_tree_node_attributes("gold_energy", id_to_gold_energy)

        return tree_nodes
