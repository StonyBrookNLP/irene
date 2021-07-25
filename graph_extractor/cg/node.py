#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import torch

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List

from transformers import AutoConfig

from cg.op_counter import count_flops_mem_bytes


@dataclass
class OpNode:
    # op: str
    id: str  # use output node debugName
    op: str  # operation type: matmul, add, mul, div, etc.
    scope: str  # scope + id
    inputs: List[DataNode]
    outputs: List[DataNode]
    flops: int = 0  # number of operations
    mem_bytes: int = 0  # bytes of mem reads and writes

    def __repr__(self):
        in_str = "\n\t\t\t\t".join(str(in_node) for in_node in self.inputs)
        f_in_str = f'\t\t\tinputs=[\n\t\t\t\t{in_str}],\n' if self.inputs else ''
        out_str = "\n\t\t\t\t".join(str(out_node) for out_node in self.outputs)
        f_out_str = f'\t\t\toutputs=[\n\t\t\t\t{out_str}],' if self.outputs else ''
        flops_str = f'flops={self.flops}, ' if self.flops else ''
        mem_str = f'mem_bytes={self.mem_bytes}, ' if self.mem_bytes else ''
        scope_str = f'scope={self.scope}, ' if self.scope else ''
        return (f'\t\t{self.__class__.__name__}(id={self.id}, op={self.op}, '
                f'{flops_str}{mem_str}{scope_str}\n'
                f'{f_in_str}{f_out_str}'
                f')')


@dataclass
class DataNode:
    id: str
    dtype: str
    shape: List[int]  # tensor shape
    extra: Dict[str, Any] = field(default_factory=dict)  # extra information

    # params: bool = False  # flattened array weights
    # mem_read_bytes: int = 0  # amount of data read from input nodes
    # mem_write_bytes: int = 0  # amount of data written to output nodes
    def __repr__(self):
        shape_str = f', shape={self.shape}' if self.shape else ''
        extra_str = f', extra={self.extra}' if self.extra else ''
        return (f'{self.__class__.__name__}(id={self.id}, dtype={self.dtype}'
                f'{shape_str}{extra_str})')


@dataclass
class Module:
    id: str  # scope + id
    nodes: List[OpNode]


@dataclass
class Graph:
    name: str
    nodes: List[OpNode]
    inputs: List[DataNode]
    outputs: List[DataNode]

    # attr: dict[str, Any] = field(default_factory=dict)  # extra information
    def __repr__(self):
        inputs_str = "\n\t\t".join(str(in_node) for in_node in self.inputs)
        in_str = f'\tinputs=[\n\t\t{inputs_str}],\n' if self.inputs else ''
        outputs_str = "\n\t\t".join(str(out_node) for out_node in self.outputs)
        o_str = f'\toutputs=[\n\t\t{outputs_str}],\n' if self.outputs else ''
        nodes_str = "\n".join(str(node) for node in self.nodes)
        n_str = f'\tnodes=[\n{nodes_str}]\n' if self.nodes else ''
        return (f'{self.__class__.__name__}(name={self.name},\n'
                f'{in_str}{o_str}{n_str}'
                f')\n')


def _process_data_nodes(node_inputs, data_nodes):
    # ListType
    data_types = set()
    in_nodes = []
    for ni in node_inputs:
        name = ni.debugName()
        node_type = ni.type()
        data_types.add(str(node_type))
        extra = dict()
        if isinstance(node_type, torch.TensorType):
            dtype = ni.type().scalarType()  # Float
            shape = ni.type().sizes()  # list of int
            # check how to detect if this is a parameter data node
        elif isinstance(node_type, torch.StringType):
            dtype = str(node_type)
            shape = []
            extra['str'] = ni.toIValue()
            # fixme: for einsum (like albert), need to expand ListConstruct
        elif isinstance(node_type, torch.IntType):
            dtype = 'int'
            shape = []
            extra['val'] = ni.toIValue()
        elif isinstance(node_type, torch.ListType):
            dtype = node_type.getElementType()
            shape = []
            # most of them are from prim::ListConstruct
            # only handle tensor here,
            for nii in ni.node().inputs():
                # only care about tensors
                if isinstance(dtype, torch.TensorType):
                    dtype = nii.type().scalarType()
                    shape.append(nii.type().sizes())
        else:
            # for types like int, int[], Long(), Device, bool, None
            # most of them are functional args, no special handling
            dtype = str(node_type)
            shape = []
        ni_node = data_nodes.get(name, DataNode(name, dtype, shape, extra))
        in_nodes.append(ni_node)
        data_nodes[name] = ni_node
    return in_nodes, data_types


def construct_aggregation_graph(trace_graph, model_name):
    # fc_input_nodes = {i.debugName(): i for i in trace_graph.inputs()}
    # fc_output_nodes = {i.debugName(): i for i in trace_graph.outputs()}
    data_nodes = dict()
    nodes = []
    gi_nodes = []
    go_nodes = []
    op_data_types = set()
    for io_node in trace_graph.inputs():
        name = io_node.debugName()
        node_type = io_node.type()
        if isinstance(io_node.type(), torch.TensorType):
            dtype = io_node.type().scalarType()  # Float
            shape = io_node.type().sizes()  # list of int
            ni_node = DataNode(name, dtype, shape)
        else:
            ni_node = DataNode(name, node_type, [])
        gi_nodes.append(ni_node)
        data_nodes[name] = ni_node

    for node in trace_graph.nodes():
        node_inputs = list(node.inputs())
        node_outputs = list(node.outputs())
        node_id = node_outputs[0].debugName()
        node_scope = node.scopeName().replace('__module.', '').split('/')[-1]
        node_op = node.kind()
        # ops.add(node_op)
        # todo: handle prim nodes: Constant, GetAttr, ListConstruct, ListUnpack,
        #  TupleConstruct, TupleUnpack, NumToTensor
        #  shrink node depth
        in_nodes, in_dtypes = _process_data_nodes(node_inputs, data_nodes)
        out_nodes, out_dtypes = _process_data_nodes(node_outputs, data_nodes)
        op_data_types.update(in_dtypes)
        op_data_types.update(out_dtypes)
        # todo: build nodes connection, set scope to modules
        op_node = OpNode(node_id, node_op, node_scope, in_nodes, out_nodes)
        op_flops, op_mem_bytes = count_flops_mem_bytes(op_node)
        op_node.flops = op_flops
        op_node.mem_bytes = op_mem_bytes
        nodes.append(op_node)
    # todo: handle return node to make it go_node
    return Graph(model_name, nodes, gi_nodes, go_nodes), op_data_types


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


if __name__ == '__main__':
    import torch
    from transformers import BertModel

    # from transformers.modeling_bert import BertIntermediate
    model_name = "prajjwal1/bert-tiny"
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_act = 'gelu_fast'
    config.torchscript = True
    model = BertModel(config)
    inputs = torch.randint(1000, size=(1, 100)).long()
    trace = torch.jit.trace(model, inputs)
    # torch.jit.save(trace, "traced_bert-tiny.pt")

    fc_graph = trace.inlined_graph
    graph, graph_ops = construct_aggregation_graph(fc_graph, model_name)

    # TODO:
    #  - simplify node edges with no data_nodes
    #  - count op types, op counts for scope, data shape
    #  - count flops and mem
    # print(graph)
    max_o = 0
    sn = 0
    tn = 0
    for i, n in enumerate(graph.nodes, 1):
        max_o = max(max_o, len(n.outputs))
        tn += 1
        if n.scope:
            sn += 1
            print(i, n.scope)
    print({k: k.replace("::", "_") for k in sorted(graph_ops)}, max_o, sn, tn)

    # node_attr = dict(style='filled',
    #                  shape='box',
    #                  align='left',
    #                  fontsize='12',
    #                  ranksep='0.1',
    #                  height='0.2')
    # from graphviz import Digraph
    # dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    # scope_nodes = defaultdict(list)  # scope to nodes map
    # # import pygtrie
    # # scope_trie = pygtrie.StringTrie(separator='.')
    # for node in graph.nodes:
    #     op = node.op
    #     scope = node.scope
    #     if not scope:
    #         # no scope nodes belong to the root graph
    #         dot.node(node.id, label=op)
    #         for inp in node.inputs:
    #             dot.edge(inp.id, node.id)
    #     else:
    #         # if scope_trie.has_key(scope):
    #         #     scope_nodes = scope_trie[scope]
    #         #     scope_nodes.append(node)
    #         #     scope_trie[scope] = scope_nodes
    #         # else:
    #         #     scope_trie[scope] = [node]
    #         # scope_trie[scope]
    #         scope_nodes[scope].append(node)
    # for scope, nodes in scope_nodes.items():
    #     sg = Digraph('cluster_' + scope)
    #     print('build nodes and edges', scope)
    #     for n in nodes:
    #         sg.node(n.id, label=scope + '-' + n.op)
    #         for inp in n.inputs:
    #             sg.edge(inp.id, n.id)
    #     sg.body.append(f'label="{scope}"')
    #     dot.subgraph(sg)
    # resize_graph(dot)
    # dot.render('bert-tiny.gv', view=True)
