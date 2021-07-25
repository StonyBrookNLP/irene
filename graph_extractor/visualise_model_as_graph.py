#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"

import uuid
import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import AutoModel
from cg.node import construct_aggregation_graph
from run_level_exp import construct_aggregation_graph
from graphviz import Digraph
import copy
from functools import lru_cache

class TreeNode(object):
    def __init__(self, scope, instance_type, level, parent_name, callable_module):
        self.id = uuid.uuid4().hex
        self.instance_type = instance_type
        self.scope = scope
        self.level = level
        self.parent_name = parent_name
        self.child_nodes = []
        self.callable_module = callable_module

        self.features = None
        self.model_name = None # pre viz-deadline added.
        self.batch_size = None
        self.seq_len = None
        self.gold_energy = None
        self.predicted_energy = None
        self.loss = None
        self.subtree_loss = None
        self.subtree_error_sum = None
        self.subtree_error_count = None
        self.subtree_error_perc = None

    def __str__(self):
        ret = "(" + self.scope.split('.')[-1]
        for child in self.child_nodes:
            ret += "," + child.__str__()
        ret += ")"
        return ret

    def description(self):
        print(f'NODE INFORMATION - Scope: {self.scope}, instance type: {self.instance_type}, level: {self.level}, parent: {self.parent_name}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        help='Huggingface model name to work with '
                             '(e.g. bert-base-uncased, distilbert-base-uncased,'
                             'google/mobilebert-uncased, roberta-base,'
                             'bert-large-uncased, roberta-large,'
                             'xlnet-base-cased, xlnet-large-cased,'
                             'albert-base-v2, albert-large-v2, t5-small, t5-base,'
                             'openai-gpt, gpt2, sshleifer/tiny-gpt2, distilgpt2'
                             'sshleifer/tiny-ctrl, facebook/bart-base, facebook/bart-large,'
                             'sshleifer/distilbart-xsum-6-6, valhalla/distilbart-mnli-12-3',
                        default='bert-base-uncased')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size of input to run model with', default=1)
    parser.add_argument('--input-len', type=int,
                        help='Length of input to run model with', default=100)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true',
                        help='Remove use of CUDA')
    parser.add_argument('--out-file', type=str,
                        help='Graphviz representation file name')
    args, _ = parser.parse_known_args()
    return args

@lru_cache(maxsize=1024)
def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    for parameters in model.parameters():
    	parameters.requires_grad_(False)
    return model

def graphviz_representation(tree):

    """
    This function creates a graphviz style digraph representation for the model graph
    """

    math_ops_name = ['matmul', 'bmm', 'softmax', 'einsum']
    ml_ops_name = ['Linear', 'LayerNorm', 'Embedding', 'BatchNorm1d', 'Conv1d', 'MaxPool1d', 'AvgPool1d', 'LSTM', 'Tanh', 'Conv1D', 'LogSigmoid', 'ReLU', 'Sigmoid', 'GELU', 'LeakyReLU']

    dot = Digraph(comment='Model Graph',
                  graph_attr=dict(rankdir='LR'))
    node_count = 0
    graphviz_node_id_mapping = {}
    # first create nodes with their labels
    for key, node in tree.items():
        node_name = node.scope.split('.')[-1]
        node_suffix = f':{node_name}' if node_name.isnumeric() else ''

        if node.instance_type in ml_ops_name:
            dot.attr('node', style='filled', fillcolor='#F4D1AE', fontsize='22',
                     color='orange', shape='rectangle')
            node_suffix = f':{node_name}'
        elif node.instance_type in math_ops_name:
            dot.attr('node', style='filled', fillcolor='#E1C9B2', fontsize='22',
                     color='#CD887D', shape='rectangle')
        else:
            dot.attr('node', style='filled', fillcolor='#DCE9F2', fontsize='22',
                     color='#007AC5', shape='oval')
        node_label = node.instance_type + node_suffix
        dot.node(str(node_count), node_label)
        graphviz_node_id_mapping[node.scope] = str(node_count)
        node_count += 1
    # add edges between nodes using node ids assigned in previous loop
    for key, node in tree.items():
        for child_node in node.child_nodes:
            dot.edge(graphviz_node_id_mapping[node.scope],
                     graphviz_node_id_mapping[child_node.scope],
                     arrowsize='.5', weight='2.')
    return dot

@lru_cache(maxsize=1024)
def create_tree_from_modules(model):

    """
    Use the named modules of a model to create a Newick format tree to visualise all the parts of a model
    """

    model_operation_information = []

    # create an iterable list of module information since model.named_modules does not function as a true list
    module_list_scope_names = []
    for (name, module) in model.named_modules():
        mname = module.__class__.__name__
        if name == '':
            prefix = 'root'
        elif mname == 'ModuleList':
            module_list_scope_name = name.split('.')[-1]
            module_list_scope_names.append(module_list_scope_name)
            continue
        else:
            prefix = 'root.'
        tmp_scope_name = prefix+name
        scope_name = '.'.join([arg for arg in tmp_scope_name.split('.') if arg not in module_list_scope_names])
        model_operation_information.append((prefix+name, mname, module))

    # add prefix to every node's scope since, by default, the root node of a graph in PyTorch has empty string as scope
    for operations in model_operation_information:
        scope = operations[0]
        mname = operations[1]
        module = operations[2]
        if scope == 'root':
            parent_name = ''
        elif mname == 'ModuleList':
            continue
        else:
            scope_name = '.'.join([arg for arg in scope.split('.') if arg not in module_list_scope_names])
            parent_name = '.'.join(scope_name.split('.')[:-1])

    root_operation = model_operation_information[0]
    root = TreeNode('root', root_operation[1], 0, '', operations[2])

    tree = {}
    tree['root'] = root

    # create nodes and keep track of parent-child relationships
    parent_child_nodes = defaultdict(list)
    for operations in model_operation_information[1:]:
        scope = operations[0]
        scope = '.'.join([arg for arg in scope.split('.') if arg not in module_list_scope_names])
        instance_type = operations[1]
        if instance_type == 'Dropout':
            continue
        if instance_type == 'ModuleList':
            continue
        module = operations[2]
        level = len(scope.split('.')) - 1
        parent_name = '.'.join(scope.split('.')[:-1])
        node = TreeNode(scope, instance_type, level, parent_name, module)
        parent_child_nodes[parent_name].append(node)
        tree[scope] = node

    # add child information to each node using the information previously stored about parent-child relationships
    for name, node in tree.items():
        node.child_nodes = parent_child_nodes[name]

    return root, tree, module_list_scope_names

@lru_cache(maxsize=1024)
def run_model_to_graph(model_name, device):
    model = load_model(model_name)
    inputs = torch.randint(1000, size=(8, 32)).long()
    model = model.eval().to(device)
    inputs = inputs.to(device)

    trace = torch.jit.trace(model, inputs)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, model_name)

    root, tree, module_list_scope_names = create_tree_from_modules(model)

    # these dictionaries are important when reconciling jit trace to the tree created from modules
    id_to_node_map = dict()
    scope_to_node_ids_map = defaultdict(list)
    for node in graph.nodes:
        id_to_node_map[node.id] = node
        node_scope = node.scope
        if node.scope == '':
            node_scope = 'root'
        scope_name = '.'.join([arg for arg in node_scope.split('.') if arg not in module_list_scope_names])
        scope_to_node_ids_map[scope_name].append(node.id)

    # for math ops, look into jit trace, create nodes accordingly and add them to correct position in model graph
    for scope, node_ids in scope_to_node_ids_map.items():
        for node_id in node_ids:
            jit_node = id_to_node_map[node_id]
            if jit_node.mem_bytes != 0 or jit_node.flops != 0:
                if jit_node.scope == '':
                    scope_to_match = 'root'
                else:
                    scope_to_match = 'root.' + jit_node.scope
                scope_to_match = '.'.join([arg for arg in scope_to_match.split('.') if arg not in module_list_scope_names])
                if jit_node.op not in ['aten::matmul', 'aten::bmm', 'aten::einsum', 'aten::softmax']:
                    continue
                node_in_position = tree[scope_to_match]
                if node_in_position.instance_type in ['Linear', 'LayerNorm', 'Embedding', 'BatchNorm1d', \
                       'Conv1d', 'MaxPool1d', 'AvgPool1d', 'LSTM', 'Tanh', \
                       'Conv1D', 'LogSigmoid', 'ReLU', 'Sigmoid', 'GELU', 'LeakyReLU']:
                    continue
                new_scope_name = scope_to_match + '.' + jit_node.op.split('::')[-1]
                new_node = TreeNode(new_scope_name, jit_node.op.split('::')[-1], node_in_position.level+1, node_in_position.scope, jit_node.op)
                node_in_position.child_nodes.append(new_node)
                tree[new_node.scope] = new_node
    return root, tree, module_list_scope_names

def main(args):
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not args.no_cuda else "cpu")

    root, tree, _ = run_model_to_graph(args.model_name, device)

    dot = graphviz_representation(tree)

    # save graphviz representation to file, then use shell command to generate final graph image
    out_file = Path(args.out_file)
    out_dir = out_file.parent
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w+') as fp:
        fp.write(dot.source)

    # in shell, run ```dot -Tpdf args.out_file -o final_graph_file.pdf``` to finally generate graph from source code saved above

if __name__ == '__main__':
    args = parse_args()
    main(args)
