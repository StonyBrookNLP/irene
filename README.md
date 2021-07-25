
# IrEne: Interpretable Energy Prediction for Transformers

This repository contains associated data and code for our [ACL'21 paper](https://arxiv.org/pdf/2106.01199.pdf). 

> **Disclaimer**: This is not the original code we used in the paper. We've cleaned up the code and standardized our dataset format for extensibility and usability. Retraining the models with new code and data format doesn't lead to exactly the same results, but they are very close. If you you want to reproduce our original results identically, please check `original` branch and instructions.


## Installation

```
conda create -n irene python=3.7 -y && conda activate irene
pip install -r requirements.txt
```

## IrEne Data

IrEne data consists of energy measurement information from 6 transformer models, each with various batch-size and sequence_length combinations. IrEne represents transformer models in a tree-based abstraction, and contains measured energy and relevant features for each node of the tree. The collected data is available for two measurement devices in `datasets/device_1.jsonl`, and `datasets/device_2.jsonl`.

Each json line in the above files correspond to certain transformer model when run with some batch size and sequence length. The root of the json has this information in the following format:

```python
{
    "model_name": "<str>", # Eg. roberta-base
    "batch_size": "<int>",
    "seq_len": "<int>",
    "frontend_tree": {
        # /* This is a nested tree explained below */
    }
}
```
The `frontend_tree` is a nested tree, where nodes are represented in the following json format:

```python
{
    "id": "<str>", # (Universally) unique identifier for the node.
    "scope": "<str>", # This is the path to the pytorch module/operation (eg. root.pooler.activation)
    "parent_name": "<str>", # Scope of the parent node.
    "level": 2, # Depth of tree at which the node exists.
    "instance_type": "Embedding", # python class name of the model/module/operation (eg. Embedding, BertModel etc)
    "type": "<str>", # Type of the node (It can be model, module or ml). See paper for details.

    # features start (See paper for more details)
    "num_parameters": "<int>",
    "flops": "<int>",
    "mem_bytes": "<float>",
    "cpu": "<float>",
    "mem": "<int>",
    "gpu": "<int>",
    "gpu_mem": "<int>",
    "gpu_clk": "<int>",
    "gpu_mem_clk": "<int>",
    "times_mean": "<float>",
    "gpu_energy_mean": "<float>",
    # features end

    # Energies
    "ground_truth_energy": "<float>", # Measured energy in Joules.
    # "predicted_energy": "<float>" # This key is won't be present, evaluation expects this to be filled in for each node.

    # Children Info
    "child_nodes_obj": [ # This will be empty for leaf nodes (ml nodes)
        {
            # This is a child dict with same fields as above
        },
        {
            # ..
        },
    ],
    "child_nodes": ["<str>", "<str>"], # scopes (operation/module paths) of the children.
}
```

## IrEne Evaluation

If you have made a new predictive model for IrEne data and want to evaluate it, populate the `predicted_energy` of each node in each json and save it in the same jsonl format, and run the following evaluation script. It'll give you average percentage errors for each type of nodes (`ml`, `module`, `model`).

```bash
python evaluate.py /path/to/original_data.jsonl /path/to/predictions_data.jsonl

# Example output:
# {
#     "ml": 0.6,
#     "module": 8.34,
#     "model": 3.78
# }
```

If you want to use IrEne predictive model, see below.


## Training, Predicting and Evaluating IrEne Model

To train a model with default (IrEne) configs, just run:

```bash
python train.py datasets/irene_device_1.jsonl serialization_directory/irene_device_1
#               ^ /path/to/irene_data.jsonl   ^ /directory/to/save/model
```

By default, it takes the following config:
```python
# Default IrEne Model Config (used in the paper)
{
    "feature_sets": ["model_specs", "resource_utilization"], # feature groups to use.
    "polynomial_features": true, # whether to consider polyomial feature interaction or not.
    "training_strategy": "end2end", # Options: "end2end", "stepwise", "unstructured", "none"
    "standardize_features": true, # whether to scale normalize features
    "tanh_scalar": 10, # Tau from equation 3 in the paper.
    "normalize_loss_scale": true, # whether to recale loss based on the scale of the nodes ground-truth energy.
    "weighting_setting": "close_to_one", # Options: "exact_one", "close_to_one", "free"
}
```
but you can change it by passing `--config_filepath /path/to/my_config.json`.

See **TODO-docstring-link** for more explanation of these configs.

Once you've the trained model, you can generate predictions as follows:

```bash
python predict.py datasets/irene_device_1.jsonl  irene_device_1_predictions.jsonl serialization_directory/irene_device_1
#                 ^ /data/to/predict/on          ^ /path/to/save/predictions      ^ /path/to/saved/model/dir
#
# (it's trained and tested on same dataset just for an example)
```

Finally, you can evaluate the generated predictions with:

```bash
python evaluate.py datasets/irene_device_1.jsonl irene_device_1_predictions.jsonl
#                  ^ /path/with/ground-truths    ^ /path/to/saved/predictions

# Output:
# Percentage Error Results:
# {
#     "ml": 0.56,
#     "module": 8.36,
#     "model": 3.4
# }
```


## CrossValidating IrEne Models

Since the dataset is of small size, we used cross-validation (leaving one transformer model type out) to evaluate the predictive models. You can run following cross-validation script, and it'll give you a following kind of report.

```bash
python crossvalidate.py datasets/irene_device_1.jsonl serialization_directory/irene_device_1
#                       ^ /path/to/irene_data.jsonl   ^ /path/to/saved/model/dir

# Percentage Error - Cross Validation Report
#            left-model-name  ml % error  module % error  model % error
# 0             roberta-base        0.71            5.49           6.91
# 1                     gpt2        0.63           14.92           4.88
# 2  distilbert-base-uncased        0.60            6.04          19.01
# 3               openai-gpt        0.92           14.01           2.96
# 4               distilgpt2        0.64           14.78           2.75
# 5        bert-base-uncased        0.70            5.45           3.93
# 6                  overall        0.70           10.12           6.74
```

Here again you can pass `--config_filepath` argument.


## IrEne Demo and Visualization

Want to look at interactive visualization of predicted energies of transformers? Head on to [this page](http://irene-viz-1.herokuapp.com/)!


## Citation

If you find this work useful, please cite it using:

```
@misc{cao2021irene,
   title={IrEne: Interpretable Energy Prediction for Transformers},
   author={Qingqing Cao and Yash Kumar Lal and Harsh Trivedi and Aruna Balasubramanian and Niranjan Balasubramanian},
   year={2021},
   eprint={2106.01199},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
