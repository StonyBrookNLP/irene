#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"

from typing import Optional

from cg.node import construct_aggregation_graph

"""
The 'information' variable in the script contains all the requisite information
 for one run of the model
Example usage: 
python calibrate_e_ml.py --model-name bert-base-uncased --batch-size 2 --input-len 100 --no-cuda
Arguments:
    model-name: model name in huggingface repository (e.g. prajjwal1/bert-tiny)
    batch-size: batch size of input tensor, first parameter of shape for model input
    input-len: input length of input tensor, second parameter of shape for model input
    no-cuda: use if you don't want the script to use CUDA
"""

import argparse
import time
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel
from transformers import modeling_utils

start_times = dict()
end_times = dict()
module_inputs = dict()
module_in_kwargs = dict()
module_outputs = dict()
modules = dict()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        help='Huggingface model name to work with '
                             '(e.g. bert-base-uncased, distilbert-base-uncased,'
                             'google/mobilebert-uncased, roberta-base,'
                             'bert-large-uncased, roberta-large,'
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
    parser.add_argument("-t", "--level_type", type=str, required=True,
                        choices=('ml', 'ml-np', 'module', 'model'),
                        help="ml, module, model type")
    args, _ = parser.parse_known_args()
    return args


def log_end_builder(name):
    def log_end(module, m_in, m_in_kwargs, m_out):
        # fixme: patch/mock method register_forward_hook in nn.Module
        end_times[f'{name}:{module.__class__.__name__}'] = time.perf_counter()
        module_inputs[f'{name}:{module.__class__.__name__}'] = m_in
        module_in_kwargs[f'{name}:{module.__class__.__name__}'] = m_in_kwargs
        module_outputs[f'{name}:{module.__class__.__name__}'] = m_out
        modules[f'{name}:{module.__class__.__name__}'] = module

    return log_end


def is_ml_operation(module):
    """
    This function checks if any given module is of a type that
    we want to analyse for E_ML operations
    """

    e_ml_operations = {nn.Linear, nn.LayerNorm, nn.Embedding, nn.BatchNorm1d,
                       nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, nn.LSTM, nn.Tanh,
                       modeling_utils.Conv1D, nn.LogSigmoid, nn.ReLU, nn.Sigmoid,
                       nn.GELU, nn.LeakyReLU}

    for e_ml_op in e_ml_operations:
        if isinstance(module, e_ml_op):
            return True
    return False


def is_ignore_operation(module):
    ignore_operations = {nn.Dropout, }
    for e in ignore_operations:
        if isinstance(module, e):
            return True
    return False


def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    torch.set_grad_enabled(False)
    return model


def get_all_operations(model_name):
    """
    This function returns the class names of all operations used in a model
    """

    model = load_model(model_name)

    all_operations = set()

    for (name, module) in model.named_modules():
        mname = module.__class__.__name__
        all_operations.add(mname)

    return all_operations


def is_level_module(level_type, module):
    if level_type == 'ml':
        return is_ml_operation(module)
    else:
        # this return model level
        return not is_ml_operation(module)


class MatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class BMM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


class EinSum(nn.Module):

    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def forward(self, *x):
        return torch.einsum(self.transformation, *x)


def get_non_parametric_ml_ops(model, input_ids):
    # todo: add other non-parametric ops if there are any
    #   if not a nn.module, wrap it like MatMul Module above
    non_param_scopes = {name for name, module in model.named_modules() if
                        not is_ml_operation(module)}
    trace = torch.jit.trace(model, input_ids)
    trace_graph = trace.inlined_graph
    graph, op_data_types = construct_aggregation_graph(trace_graph, 'model')
    information = defaultdict(list)
    for node in graph.nodes:
        if node.op == 'aten::softmax':
            args = node.inputs
            inputs = torch.rand(args[0].shape, dtype=torch.float,
                                device=model.device)
            module_arg = args[1].extra['val']
            module_info = {'name': f'{node.scope}:Softmax',
                           'module': nn.Softmax(module_arg),
                           'inputs': (inputs,),
                           'in_kwargs': {},
                           }
            information['softmax'].append(module_info)
        if node.op == 'aten::matmul' and node.scope in non_param_scopes:
            args = node.inputs
            inputs = [torch.rand(arg.shape, dtype=torch.float,
                                 device=model.device) for arg in args]
            module_info = {'name': f'{node.scope}:MatMul',
                           'module': MatMul(),
                           'inputs': tuple(inputs),
                           'in_kwargs': {},
                           }
            information['matmul'].append(module_info)
        if node.op == 'aten::bmm' and node.scope in non_param_scopes:
            args = node.inputs
            inputs = [torch.rand(arg.shape, dtype=torch.float,
                                 device=model.device) for arg in args if
                      arg.dtype != None and len(arg.shape) > 0]
            module_info = {'name': node.scope, 'module': BMM(),
                           'inputs': tuple(inputs),
                           'in_kwargs': {},
                           }
            information['bmm'].append(module_info)
        if node.op == 'aten::einsum' and node.scope in non_param_scopes:
            args = node.inputs
            inputs = []
            transformation = None
            for arg in args:
                if arg.dtype == 'str':
                    transformation = arg.extra['str']
                    continue
                elif arg.dtype is None:
                    continue
                elif len(arg.shape) == 0:
                    continue
                input_t = torch.rand(arg.shape[0], dtype=torch.float,
                                     device=model.device)
                inputs.append(input_t)
            module_info = {'name': node.scope,
                           'module': EinSum(transformation),
                           'inputs': tuple(inputs),
                           'in_kwargs': {},
                           }
            information['einsum'].append(module_info)
    return information


def get_module_info(model_name, batch_size, input_len, device,
                    level_type='ml'):
    model = load_model(model_name)
    model = model.eval().to(device)
    inputs = torch.randint(1000, size=(batch_size, input_len)).long()
    inputs = inputs.to(device)
    if 't5' in model_name:
        # t5 does not work for non parametric operations
        inputs = (inputs, inputs)
    if level_type == 'ml-np':
        information = get_non_parametric_ml_ops(model, inputs)
        return information

    start_times.clear()
    end_times.clear()
    module_inputs.clear()
    module_in_kwargs.clear()
    module_outputs.clear()
    modules.clear()
    for (name, module) in model.named_modules():
        if not name:
            continue
        # module.register_forward_pre_hook(log_start_builder(name))
        if is_ignore_operation(module):
            continue
        module.register_forward_hook(log_end_builder(name))

    kwargs = {'decoder_input_ids': inputs} if hasattr(model, 'decoder') else {}
    _ = model(inputs, **kwargs)

    information = defaultdict(list)
    for module_name in end_times.keys():
        if module_name not in modules:
            continue
        module = modules[module_name]
        if is_level_module(level_type, module):
            module_info = {'name': module_name, 'module': modules[module_name],
                           'inputs': module_inputs[module_name],
                           'in_kwargs': module_in_kwargs[module_name],
                           # 'outputs': module_outputs[module_name],
                           # 'runtime': end_times[module_name] - start_times[
                           # module_name]
                           }

            module_identifier = module_name.split(':')[-1]
            information[module_identifier].append(module_info)

    return information


def print_info(information):
    for k, info in information.items():
        print(k, len(info))
        infoi = info[0]
        ii = infoi['inputs']
        ii_kwargs = infoi['in_kwargs']
        if ii_kwargs:
            print(type(ii_kwargs), infoi['name'],
                  {iik: v.shape if isinstance(v, torch.Tensor) else v
                   for iik, v in ii_kwargs.items()})
        if ii:
            print(type(ii), infoi['name'],
                  [v.shape if isinstance(v, torch.Tensor) else v
                   for v in ii])

        print()


def main(args):
    operation_names = get_all_operations(args.model_name)
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not args.no_cuda else "cpu")
    information = get_module_info(args.model_name, args.batch_size,
                                  args.input_len, device, args.level_type)


if __name__ == '__main__':
    main(parse_args())
