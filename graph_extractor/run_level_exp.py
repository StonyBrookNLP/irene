#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import copy
import inspect
import json
import time
from functools import partial
from functools import update_wrapper
from pathlib import Path
from common import logger
from common import sanitize
import torch
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import get_module_info
from cg.node import construct_aggregation_graph


def get_flops_mem_bytes(graph):
    flops = dict()
    mem_bytes = dict()

    for node in graph.nodes:
        flops[node.id] = node.flops
        mem_bytes[node.id] = node.mem_bytes
        # if node.flops:
        #     print(node.op, node.flops, node.mem_bytes)
    return sum(flops.values()), sum(mem_bytes.values())


def get_model_flops_mem_bytes(module_fn, inputs, module_name, device):
    cpu_inputs = tuple([i.cpu() for i in inputs])
    trace = torch.jit.trace(module_fn.cpu(), cpu_inputs)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, module_name)
    flops, mem_bytes = get_flops_mem_bytes(graph)
    module_fn.to(device)
    tuple([i.to(device) for i in inputs])
    return flops, mem_bytes


def calibrate_repeats(fn, fi, fi_kwargs, probe_repeats):
    # probe_repeats set to 10 for ml, 5 for module, 3 for model
    start = time.perf_counter()
    needed = 0
    while True:
        for _ in range(probe_repeats):
            _ = fn(*fi, **fi_kwargs)
        end = time.perf_counter()
        needed += 1
        if end - start > 5:  # run 5 seconds
            break
    repeats = max(probe_repeats, needed * probe_repeats)
    return repeats * 4


def run_model(model_name, bs, seq_len, probe_repeats, runs, device, multi_gpu):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    model = model.eval().to(device)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    input_ids = torch.randint(1000, size=(bs, seq_len), dtype=torch.long,
                              device=device)
    inputs = (input_ids,)
    if config.model_type == 't5':
        #  attention_mask=None, decoder_input_ids=None
        inputs = (input_ids, input_ids)
    flops, mem_bytes = get_model_flops_mem_bytes(model, inputs,
                                                 model_name, device)
    model_prof = dict(name=model_name, flops=flops, mem_bytes=mem_bytes)
    repeats = calibrate_repeats(model, inputs, {}, probe_repeats)
    model_prof['repeats'] = repeats
    logger.info(f'{model_name}_b{bs}_i{seq_len}, '
                f'flops={flops}, mem_bytes={mem_bytes}, '
                f'repeats={repeats}')
    seq2seq = hasattr(model, 'decoder')
    kwargs = {'decoder_input_ids': input_ids} if seq2seq else {}
    for run in range(1, runs + 1):
        logger.info(f'run {model_name}_b{bs}_i{seq_len} ({run}/{runs})')
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(repeats):
            _ = model(input_ids, **kwargs)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        model_prof[f'start_{run}'] = level_start
        model_prof[f'end_{run}'] = level_end
        time.sleep(3)  # sleep 3s to cool down
    return model_prof


level_sigs = dict()


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def run_ml_or_module(model_name, bs, seq_len, probe_repeats, runs, device,
                     level, level_name, multi_gpu=False):
    # # uncomment support specific ML levels
    # if level_name != 'linear':
    #     return None
    fn = level['module']
    fname = level['name']
    fi = level['inputs']
    assert isinstance(fi, tuple), f'{fi} must be tuple!'
    fi_kwargs = level['in_kwargs']
    assert isinstance(fi_kwargs, dict), f'{fi_kwargs} must be dict!'
    # separate tensor args and rest from fi
    fn_fwd = fn.forward
    #  unify fi and fi_kwargs
    fn_args = inspect.getfullargspec(fn_fwd).args
    fill_args = dict()
    fi_args = fn_args[1:1 + len(fi)]
    ti = []
    for fi_k, fi_v in zip(fi_args, fi):
        if isinstance(fi_v, torch.Tensor):
            ti.append(fi_v)
        else:
            fill_args[fi_k] = fi_v
    for k, v in fi_kwargs.items():
        if isinstance(v, torch.Tensor):
            ti.append(v)
        else:
            fill_args[k] = v
    # wrap forward into traceable fn (only tensor args)
    # https://github.com/pytorch/pytorch/issues/14455#issuecomment-445962680
    fn.forward = wrapped_partial(fn.forward, **fill_args)
    flops, mem_bytes = get_model_flops_mem_bytes(fn, ti, fname, device)
    fn.forward = fn_fwd
    sig = f"{level_name},{flops},{mem_bytes}"
    level_prof = dict(name=fname, flops=flops, mem_bytes=mem_bytes)

    if sig in level_sigs:
        logger.info(f'{fname} already profiled, return value from {sig}')
        cached_prof = copy.deepcopy(level_sigs[sig])
        cached_prof['name'] = fname
        return cached_prof
    if multi_gpu:
        fn = torch.nn.DataParallel(fn)
    calibrated_repeats = calibrate_repeats(fn, fi, fi_kwargs, probe_repeats)
    level_prof['repeats'] = calibrated_repeats
    logger.info(f'{model_name}_b{bs}_i{seq_len}_{level_name}, '
                f'flops={flops}, mem_bytes={mem_bytes}, '
                f'repeats={calibrated_repeats}, {fname}')
    for run in range(1, runs + 1):
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(calibrated_repeats):
            _ = fn(*fi, **fi_kwargs)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        level_prof[f'start_{run}'] = level_start
        level_prof[f'end_{run}'] = level_end
        logger.info(f'run {model_name}_b{bs}_i{seq_len}_{level_name}, '
                    f'({run}/{runs}) {fname} done.')
        time.sleep(1)  # sleep 1s to cool down
    level_sigs[sig] = level_prof
    return level_prof


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs
    probe_repeats = args.probe_repeats

    torch.set_grad_enabled(False)
    use_cuda = not args.no_cuda
    multi_gpu = args.multi_gpu
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and use_cuda else "cpu")
    seq_len = args.input_length
    bs = args.batch_size
    level_type = args.level_type
    model_name = args.model_name
    # for model_name in args.models:
    logger.info(f'profiling {model_name} {level_type} on {device}...')
    model_prof_info = []
    model_name_s = sanitize(model_name)
    filename = f'{model_name_s}_{level_type}_r{runs}_b{bs}_i{seq_len}.json'
    prof_info_file = out_dir.joinpath(filename)
    if prof_info_file.exists():
        logger.info(f'{filename} already profiled, skip')
        return
    if level_type == 'model':
        if args.log_energy_consumption:
            from experiment_impact_tracker.compute_tracker import ImpactTracker

            logger.info("Launching impact tracker...")
            tracker = ImpactTracker(args.energy_output_dir)
            tracker.launch_impact_monitor()
        prof_info = run_model(model_name, bs, seq_len,
                              probe_repeats, runs, device, multi_gpu)
        model_prof_info.append(prof_info)
    else:
        information = get_module_info(model_name, bs, seq_len, device,
                                      level_type)
        for level_name, levels in information.items():
            for level in levels:
                prof_info = run_ml_or_module(model_name, bs, seq_len,
                                             probe_repeats, runs, device,
                                             level, level_name, multi_gpu)
                if prof_info is None:
                    continue
                prof_info['type'] = level_name
                model_prof_info.append(prof_info)
    prof_info_file.write_text(json.dumps(model_prof_info))
    logger.info(f'{model_name} done.')
    # logger.info('all done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-t", "--level_type", type=str, required=True,
                        choices=('ml', 'ml-np', 'module', 'model'),
                        help="ml, module, model type")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=384,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="iterations to run the model")
    parser.add_argument("-n", "--probe_repeats", type=int, default=10,
                        help="initial probing iterations to run the model")
    parser.add_argument("-m", "--model_name", type=str,
                        help="model string supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-nc", "--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("-le", "--log_energy_consumption", action="store_true",
                        help="Whether to track energy consumption")
    parser.add_argument("-mg", "--multi_gpu", action="store_true",
                        help="Whether to all gpus or not")
    parser.add_argument("-eo", "--energy_output_dir", type=str,
                        help="model string supported by the "
                             "HuggingFace Transformers library")
    main(parser.parse_args())
