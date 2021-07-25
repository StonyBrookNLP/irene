from typing import List, Tuple
import sys
import os

from tqdm import tqdm
import torch
from lib.tree_node import TreeNode
from lib.utils import get_percentage_error_item

torch.random.manual_seed(13370)


class WeightingModel(torch.nn.Module):
    """
    Node weighting model for the IrEne model.
    """

    def __init__(self, num_features: int, setting: str, tanh_scalar: int = 10):
        super(WeightingModel, self).__init__()
        self.num_features = num_features
        self.module = torch.nn.Linear(num_features, 1)
        assert setting in (
            "exact_one",
            "close_to_one",
            "free",
        ), f"invalid setting {setting}."
        self.setting = setting
        self.tanh_scalar = tanh_scalar

    def forward(self, features):
        if self.setting == "exact_one":
            return torch.tensor(1.0)

        elif self.setting == "close_to_one":
            return 1 + (self.module(features).tanh().squeeze() / self.tanh_scalar)

        elif self.setting == "free":
            return self.module(features).squeeze()


class RecursiveTreeComputation(torch.nn.Module):
    """
    Main PyTorch module of IrEne model.

    num_features: number of features each node is expected to have.
    weighting_setting: how to scale weights of child-node energies to combine for the parent node.
        exact_one: non-learnable and fixed as 1.
        free: freely learnable scales/weights.
        close_to_one: learnable but regularized to be close to 1.
    tanh_scalar: See taug in equation 1 in paper. Only applicable when weighting_setting is close_to_one.
    training_strategy: can be in "end2end", "stepwise", "unstructured", "none" (See paper for details.)
    normalize_loss_scale: Whether to normalize the loss of nodes based on the scales of their absolute energies.
    """

    def __init__(
        self,
        num_features: int,
        weighting_setting: str,
        tanh_scalar: float = 10,
        training_strategy: str = "end2end",
        normalize_loss_scale: bool = False,
    ):
        super(RecursiveTreeComputation, self).__init__()
        self.weighting_model = WeightingModel(
            num_features=num_features,
            setting=weighting_setting,
            tanh_scalar=tanh_scalar,
        )
        self.training_strategy = training_strategy
        if training_strategy:
            assert training_strategy in (
                "end2end",
                "stepwise",
                "unstructured",
                "none",
            ), "invalid train strategy."
        self.normalize_loss_scale = normalize_loss_scale

    def forward(self, node, training: bool = False):

        if not node.child_nodes:
            assert (
                node.predicted_energy is not None
            ), "Found leaf node w/o energy prediction."
            node.loss = node.subtree_loss = 0.0
            node.subtree_error_sum = 0.0
            node.subtree_error_count = 0
            node.subtree_error_perc = 0.0
            return node

        node.predicted_energy = 0

        node.subtree_loss = 0.0
        node.subtree_error_sum = 0.0
        node.subtree_error_count = 0

        for child_node in node.child_nodes:

            child_node = self.forward(child_node, training=training)
            child_node_score = self.weighting_model.forward(child_node.features)

            if self.training_strategy in ["none", "end2end"]:
                node.predicted_energy += child_node_score * child_node.predicted_energy

            elif self.training_strategy == "stepwise":
                if training:
                    node.predicted_energy += child_node_score * child_node.gold_energy
                else:
                    node.predicted_energy += (
                        child_node_score * child_node.predicted_energy
                    )

            # TODO: Add an option to toggle between using only root node supervision
            # vs all nodes supervision.
            node.subtree_loss += child_node.subtree_loss
            node.subtree_error_sum += child_node.subtree_error_sum
            node.subtree_error_count += child_node.subtree_error_count

        if self.training_strategy == "unstructured":
            node.predicted_energy = self.weighting_model.forward(node.features)

        if node.gold_energy:
            mse_loss_func = torch.nn.MSELoss()
            node.loss = mse_loss_func(node.predicted_energy, node.gold_energy)
            if self.normalize_loss_scale:
                node.loss = node.loss / (node.gold_energy ** 2)
            node.subtree_loss += node.loss

            node.subtree_error_sum += get_percentage_error_item(
                node.gold_energy, node.predicted_energy
            )
            node.subtree_error_count += 1
            node.subtree_error_perc = float(
                node.subtree_error_sum / node.subtree_error_count
            )

        return node


def train_non_ml_level_model(
    train_trees: List[TreeNode],
    standardize_features: bool,
    polynomial_features: bool,
    weighting_setting: str,
    tanh_scalar: float,
    training_strategy: str,
    normalize_loss_scale: bool,
) -> Tuple:
    """
    Training loop for the main model.

    train_trees: List of `TreeNode` objects to train on.
    standardize_features: whether to normalize/standardize features or not.
    polynomial_features: whether to augment features with their polynomial variations.
    weighting_setting: how to scale weights of child-node energies to combine for the parent node.
        exact_one: non-learnable and fixed as 1.
        free: freely learnable scales/weights.
        close_to_one: learnable but regularized to be close to 1.
    tanh_scalar: See taug in equation 1 in paper. Only applicable when weighting_setting is close_to_one.
    normalize_loss_scale: Whether to normalize the loss of nodes based on the scales of their absolute energies.
    """

    train_trees, non_ml_transformations = TreeNode.prepare_for_non_ml_training(
        train_trees,
        standardize_features=standardize_features,
        polynomial_features=polynomial_features,
    )

    num_features = train_trees[0].features.shape[1]
    arguments = {
        "num_features": num_features,
        "weighting_setting": weighting_setting,
        "tanh_scalar": tanh_scalar,
        "training_strategy": training_strategy,
        "normalize_loss_scale": normalize_loss_scale,
    }
    non_ml_level_model = RecursiveTreeComputation(**arguments)

    num_epochs = 8
    device = torch.device("cpu")
    non_ml_level_model = non_ml_level_model.to(device)

    optimizer = torch.optim.Adam(non_ml_level_model.parameters(), lr=1e-3)

    tqdm_obj = tqdm(range(num_epochs))
    for _ in tqdm_obj:

        total_loss = 0.0
        perc_error_sum = 0.0
        perc_error_count = 0
        for train_tree in train_trees:

            train_tree = non_ml_level_model(train_tree, training=True)
            loss = train_tree.subtree_loss
            assert loss.requires_grad
            total_loss += loss.item()

            perc_error_sum += float(train_tree.subtree_error_sum)
            perc_error_count += int(train_tree.subtree_error_count)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        perc_error_mean = round(perc_error_sum / perc_error_count, 6)
        loss_mean = round(float(total_loss / perc_error_count), 6)

        tqdm_obj.set_description(
            f"Mean Node Loss: {loss_mean} Mean Node %Error: {perc_error_mean}"
        )

    return non_ml_level_model, non_ml_transformations


def predict_non_ml_level_model(
    non_ml_level_model: torch.nn.Module,
    non_ml_level_transformations: List,
    predict_trees: List[TreeNode],
) -> List[TreeNode]:
    """
    Runs recursive tree prediction model on the predict_trees.
    """

    predict_trees = TreeNode.prepare_for_non_ml_predicting(
        predict_trees, non_ml_level_transformations
    )

    device = torch.device("cpu")
    non_ml_level_model.to(device)

    predicted_trees = []
    for predict_tree in tqdm(predict_trees):
        predicted_tree = non_ml_level_model(predict_tree, training=False)
        predicted_trees.append(predicted_tree)

    return predicted_trees
