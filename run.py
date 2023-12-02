from random import randint

import codecs
import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
#from torch_geometric.datasets import coma
from numba.cuda.cudadrv import rtapi

from transformers import GPT2Tokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2PreTrainedModel, GPT2ForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import spacy
nlp = spacy.load("en_core_web_sm")
import re
from statistics import mode
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
#from rouge.rouge_score import _recon_lcs,_split_into_words
from rouge.rouge_score import _lcs
from rouge import Rouge
rouge = Rouge()
import sys
sys.setrecursionlimit(10000)

from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients, IntegratedGradients, TokenReferenceBase

from functools import lru_cache
from operator import itemgetter


import functools
from typing import Any, Callable, List, Tuple, Union, overload, cast

from captum._utils.typing import BaselineType, Literal, ModuleOrModuleList, TargetType

from captum.log import log_usage
from torch import Tensor
from torch.nn.parallel.scatter_gather import scatter
import torchvision
from captum.attr import NoiseTunnel
from captum.attr._utils.approximation_methods import SUPPORTED_METHODS
from captum.attr._utils.common import (
    _format_input_baseline,
    _sum_rows,
    _tensorize_baseline,
    _validate_input,
)
from captum._utils.common import (
    _format_additional_forward_args,
    _format_tensor_into_tuples,
    _run_forward,
    _validate_target,
    _extract_device,
    _format_outputs,
)
import csv
import string
from collections import OrderedDict
import textwrap

device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device("cuda")
def max_abs(arr):
    if abs(max(arr)) >= abs(min(arr)):
        return max(arr)
    else:
        return min(arr)
attr_phrase_pooling = max

def is_nan(x):
    return (x != x)

def nan2str(x):
    return x if not is_nan(x) else ""

def calculate_rouge_l(x, y):
    if x == y:
        return 1
    if len(x) == 0 or len(y) == 0:
        return 0
    return rouge.get_scores(x, y)[0]["rouge-l"]["f"]


def _recon_lcs(x, y):
     
    # function to find the longest common substring
 
    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1) 
     
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:
       
        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0
 
    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():
         
        # upper right triangle of the 2D array
        for k in range(len(x)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(y) - 1, -1, -1)))
         
        # lower left triangle of the 2D array
        for k in range(len(y)):       
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                    range(len(x) - 1, -1, -1)))
 
    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))



class Dataset(Dataset):

    def __init__(self, source, target, target_attrs, tokenizer, model_type, output_type, max_length=768):

        self.tokenizer = tokenizer
        self.source = source
        self.source_ids = []
        self.source_attn_masks = []
        self.target = target
        self.target_ids = []
        self.target_attn_masks = []
        self.target_attrs = target_attrs
        #self.cls_token_locations = []
        self.model_type = model_type
        self.output_type = output_type
        
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            target_max_length = max([len(nltk.word_tokenize(t_i)) for t_i in target]) +1
    
        for s_i, t_i in zip(source, target):
            if output_type == "one-hot":
                if model_type.startswith("gpt2"):
                    source_encodings_dict = tokenizer('<|startoftext|>'+ s_i + '<|endoftext|>', truncation=True, max_length=min(768, max_length), padding="max_length")
                elif model_type.startswith("bert"):
                    source_encodings_dict = tokenizer(s_i, truncation=True, padding="max_length")
                else:
                    raise NotImplementedError()
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    source_encodings_dict = tokenizer(s_i, truncation=True, max_length=max_length, padding="max_length")
                    target_encodings_dict = tokenizer(t_i, truncation=True, max_length=target_max_length, padding="max_length")
                elif model_type.startswith("gpt2"):
                    source_encodings_dict = tokenizer('<|startoftext|>'+ s_i + '\n' + t_i + '<|endoftext|>', truncation=True, max_length=min(768, max_length), padding="max_length")
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            #print(tokenizer.decode(encodings_dict['input_ids']))
            #self.cls_token_locations.append(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            #print(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            
            self.source_ids.append(torch.tensor(source_encodings_dict['input_ids']))
            self.source_attn_masks.append(torch.tensor(source_encodings_dict['attention_mask']))
            
            if model_type.startswith("t5") or model_type.startswith("google/t5"):
                self.target_ids.append(torch.tensor(target_encodings_dict['input_ids']))
                self.target_attn_masks.append(torch.tensor(target_encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx):
        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
            return self.source_ids[idx], self.source_attn_masks[idx], self.target_ids[idx], self.target_attn_masks[idx], self.target_attrs[idx]
        elif self.model_type.startswith("bert") or self.model_type.startswith("gpt2"):
            return self.source_ids[idx], self.source_attn_masks[idx], self.target[idx], self.target_attrs[idx]
        else:
            raise NotImplementedError()


class T5IntegratedGradients(LayerIntegratedGradients):
    
    def __init__(
        self,
        forward_func: Callable,
        embedding_layer: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        LayerIntegratedGradients.__init__(self, forward_func, embedding_layer, device_ids, multiply_by_inputs)
        
        
        
    def generate_reference_layer(self, input_ids, tokenizer, baseline_method, attr_mask):
        if baseline_method == "pad":
            '''
            token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
            reference_ids = token_reference.generate_reference(seq_len, device=device)
            eos_idx = input_ids.squeeze().detach().cpu().tolist().index(tokenizer.eos_token_id)
            reference_ids[eos_idx] = tokenizer.eos_token_id
            reference_layer = self.layer(reference_ids.unsqueeze(0))
            '''
            reference_ids = torch.tensor(input_ids)
            reference_ids[:,attr_mask] = tokenizer.pad_token_id
            reference_layer = self.layer(reference_ids)
            return reference_layer
        elif baseline_method == "zero":
            reference_layer = self.layer(input_ids)
            reference_layer[:,attr_mask,:] = 0
            return reference_layer
        elif baseline_method == "gaussblur":
            reference_layer = self.layer(input_ids)
            blurrer = torchvision.transforms.GaussianBlur([19, 19], sigma=(9.5, 10.0))
            reference_layer[:,attr_mask,:] = blurrer(reference_layer[:,attr_mask,:])
            return reference_layer
        else:
            raise NotImplementedError()
        
        
    @log_usage()
    def compute_convergence_delta(
        self,
        attributions: Union[Tensor, Tuple[Tensor, ...]],
        start_point: Union[
            None, int, float, Tensor, Tuple[Union[int, float, Tensor], ...]
        ],
        end_point: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
    ) -> Tensor:
        r"""
        Here we provide a specific implementation for `compute_convergence_delta`
        which is based on a common property among gradient-based attribution algorithms.
        In the literature sometimes it is also called completeness axiom. Completeness
        axiom states that the sum of the attribution must be equal to the differences of
        NN Models's function at its end and start points. In other words:
        sum(attributions) - (F(end_point) - F(start_point)) is close to zero.
        Returned delta of this method is defined as above stated difference.
        """
        end_point, start_point = _format_input_baseline(end_point, start_point)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # tensorizing start_point in case it is a scalar or one example baseline
        # If the batch size is large we could potentially also tensorize only one
        # sample and expand the output to the rest of the elements in the batch
        start_point = _tensorize_baseline(end_point, start_point)

        attributions = _format_tensor_into_tuples(attributions)

        # verify that the attributions and end_point match on 1st dimension
        for attribution, end_point_tnsr in zip(attributions, end_point):
            assert end_point_tnsr.shape[0] == attribution.shape[0], (
                "Attributions tensor and the end_point must match on the first"
                " dimension but found attribution: {} and end_point: {}".format(
                    attribution.shape[0], end_point_tnsr.shape[0]
                )
            )

        num_samples = end_point[0].shape[0]
        _validate_input(end_point, start_point)
        _validate_target(num_samples, target)

        with torch.no_grad():
            def start_point_forward_hook(
                    module, hook_inputs, hook_outputs=None, layer_idx=0
                ):
                if np.array_equal(hook_outputs.size(), start_point[0].size()):
                    return start_point[0]
                else:
                    return hook_outputs
            
            hooks = []
            try:

                layers = self.layer
                if not isinstance(layers, list):
                    layers = [self.layer]

                for layer_idx, layer in enumerate(layers):
                    hook = None
                    hook = layer.register_forward_hook(
                        functools.partial(
                            start_point_forward_hook, layer_idx=layer_idx
                        )
                    )

                    hooks.append(hook)
            
                start_out_sum = _sum_rows(
                    _run_forward(
                        self.forward_func, tuple(), target, additional_forward_args
                    )
                )
                
            finally:
                for hook in hooks:
                    if hook is not None:
                        hook.remove()    

            def end_point_forward_hook(
                    module, hook_inputs, hook_outputs=None, layer_idx=0
                ):
                
                if np.array_equal(hook_outputs.size(), end_point[0].size()):
                    return end_point[0]
                else:
                    return hook_outputs
            
            hooks = []
            try:

                layers = self.layer
                if not isinstance(layers, list):
                    layers = [self.layer]

                for layer_idx, layer in enumerate(layers):
                    hook = None
                    hook = layer.register_forward_hook(
                        functools.partial(
                            end_point_forward_hook, layer_idx=layer_idx
                        )
                    )

                    hooks.append(hook)

                end_out_sum = _sum_rows(
                    _run_forward(
                        self.forward_func, tuple(), target, additional_forward_args
                    )
                )
            finally:
                for hook in hooks:
                    if hook is not None:
                        hook.remove()        
                
            row_sums = [_sum_rows(attribution) for attribution in attributions]
            attr_sum = torch.stack(
                [cast(Tensor, sum(row_sum)) for row_sum in zip(*row_sums)]
            )
            _delta = attr_sum - (end_out_sum - start_out_sum)
            
            print("attr_sum:", attr_sum.item())
            print("start:", start_out_sum.item())
            print("end:", end_out_sum.item())
            print("end - start:", (end_out_sum - start_out_sum).item())
        return _delta
    
        
    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baseline_method,
        attr_mask, 
        tokenizer, 
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        smooth_grad: bool = False,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        inps, _ = _format_input_baseline(inputs, None)
        #_validate_input(inps, zeros, n_steps, method)
        
        assert (
            n_steps >= 0
        ), "The number of steps must be a positive integer. " "Given: {}".format(n_steps)
    
        assert (
            method in SUPPORTED_METHODS
        ), "Approximation method must be one for the following {}. " "Given {}".format(
            SUPPORTED_METHODS, method
        )

        #zeros = _tensorize_baseline(inps, zeros)
        
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        def flatten_tuple(tup):
            return tuple(
                sum((list(x) if isinstance(x, (tuple, list)) else [x] for x in tup), [])
            )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)

        
        inputs_layer = self.layer(inps[0])

        # if we have one output
        if not isinstance(self.layer, list):
            inputs_layer = (inputs_layer,)

        num_outputs = [1 if isinstance(x, Tensor) else len(x) for x in inputs_layer]
        num_outputs_cumsum = torch.cumsum(
            torch.IntTensor([0] + num_outputs), dim=0  # type: ignore
        )
        
        inputs_layer = flatten_tuple(inputs_layer)

        baselines_layer = self.generate_reference_layer(inputs, tokenizer, baseline_method, attr_mask)
        #baselines_layer = self.layer(baselines[0])
        # if we have one output
        if not isinstance(self.layer, list):
            baselines_layer = (baselines_layer,)
        baselines_layer = flatten_tuple(baselines_layer)

        # inputs -> these inputs are scaled
        def gradient_func(
            forward_fn: Callable,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target_ind: TargetType = None,
            additional_forward_args: Any = None,
        ) -> Tuple[Tensor, ...]:
            
            if self.device_ids is None or len(self.device_ids) == 0:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type:ignore
                    inputs, target_gpus=self.device_ids
                )

            scattered_inputs_dict = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }
            
            with torch.autograd.set_grad_enabled(True):
                
                def layer_forward_hook(
                    module, hook_inputs, hook_outputs=None, layer_idx=0
                ):
                    
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    is_layer_tuple = (
                        isinstance(hook_outputs, tuple)
                        # hook_outputs is None if attribute_to_layer_input == True
                        if hook_outputs is not None
                        else isinstance(hook_inputs, tuple)
                    )

                    if is_layer_tuple:
                        rt = scattered_inputs_dict[device][
                            num_outputs_cumsum[layer_idx] : num_outputs_cumsum[
                                layer_idx + 1
                            ]
                        ]
                    else:
                        rt = scattered_inputs_dict[device][num_outputs_cumsum[layer_idx]]

                    if np.array_equal(hook_outputs.size(), rt.size()):
                        return rt
                    else:
                        return hook_outputs

                hooks = []
                try:

                    layers = self.layer
                    if not isinstance(layers, list):
                        layers = [self.layer]

                    for layer_idx, layer in enumerate(layers):
                        hook = None
                        # TODO:
                        # Allow multiple attribute_to_layer_input flags for
                        # each layer, i.e. attribute_to_layer_input[layer_idx]
                        if attribute_to_layer_input:
                            hook = layer.register_forward_pre_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )
                        else:
                            hook = layer.register_forward_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )

                        hooks.append(hook)
                
                    output = _run_forward(
                        self.forward_func, tuple(), target_ind, additional_forward_args
                    )
                
                finally:
                    for hook in hooks:
                        if hook is not None:
                            hook.remove()
                
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(torch.unbind(output), inputs)
            return grads

        self.ig.gradient_func = gradient_func
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )
        if smooth_grad:
            self.nt = NoiseTunnel(self.ig)
            attributions = self.nt.attribute.__wrapped__(  # type: ignore
                self.nt,  # self
                inputs_layer,
                baselines=baselines_layer,
                target=target,
                additional_forward_args=all_inputs,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
                nt_type='smoothgrad',nt_samples=16, stdevs=0.2
            )
        else:
            attributions = self.ig.attribute.__wrapped__(  # type: ignore
                self.ig,  # self
                inputs_layer,
                baselines=baselines_layer,
                target=target,
                additional_forward_args=all_inputs,
                n_steps=n_steps,
                method=method,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=False,
            )

        # handle multiple outputs
        output: List[Tuple[Tensor, ...]] = [
            tuple(
                attributions[
                    int(num_outputs_cumsum[i]) : int(num_outputs_cumsum[i + 1])
                ]
            )
            for i in range(len(num_outputs))
        ]

        if return_convergence_delta:
            
            start_point, end_point = baselines_layer, inputs_layer
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=all_inputs,
                target=target,
            )
            return _format_outputs(isinstance(self.layer, list), output), delta

        return _format_outputs(isinstance(self.layer, list), output)

class Experiment():
    def __init__(self, seed={"preprocess": 4, "train": 4}, modality="spending+text+image", model_type="t5-base", output_type="text", epochs=25):
        #hyper-parameters
        self.seed = seed

        self.business_units = ['iTrade']
        self.modality = modality
        self.output_type = output_type # "text" "one-hot"
        self.model_type = model_type # "t5-base" "bert-base-uncased" "gpt2" "google/t5-v1_1-base"
        self.text_format = "score+diff+orig"  # "orig" "diff" "score+diff+orig"
        self.image_format = "score+diff+orig" # "orig" "score+diff+orig"
        self.k = 5  # top-k pages
        self.batch_size = 8
        self.num_labels = 3
        self.epochs = epochs
        self.learning_rate = 1e-4
        self.warmup_steps = 1e2
        self.epsilon = 1e-8
        self.num_weeks = 2
        self.seq_len = 1280
        
        self.dataset_split = [0.7, 0.10, 0.20]
        self.attr_method = 'gausslegendre' # 'gausslegendre' 'riemann_left' 'riemann_right' 'riemann_middle' 'riemann_trapezoid'
        self.baseline_method = 'gaussblur' # 'pad'
        self.spending_percentiles = [0.2,0.4,0.6,0.8] #[0.14286, 0.28571, 0.42857, 0.57143, 0.71429, 0.85714]  [0.3333,0.6667]  [0.2,0.4,0.6,0.8]
        self.funnel_percentiles = [0.3333,0.6667]
        self.smooth_grad = False
        self.attr_batch_size = 8
        self.attr_steps = 300
        self.attr_prompts = False
        self.attr_word_viz = "scale"
        self.attr_decode_subwords = True
        self.attr_delta_valid_range = [-0.190023642, 0.28805191]
    
    def value2scale(self, v, p, mode = "text"):
        if len(p)==6:
            if v<p[0]:
                if mode == "one-hot":
                    return 0
                elif mode == "text":
                    return 'extremely low'
                else:
                    raise NotImplementedError()
            elif v<p[1]:
                if mode == "one-hot":
                    return 1
                elif mode == "text":
                    return 'very low'
                else:
                    raise NotImplementedError()
            elif v<p[2]:
                if mode == "one-hot":
                    return 2
                elif mode == "text":
                    return 'low'
                else:
                    raise NotImplementedError()
            elif v<p[3]:
                if mode == "one-hot":
                    return 3
                elif mode == "text":
                    return 'medium'
                else:
                    raise NotImplementedError()
            elif v<p[4]:
                if mode == "one-hot":
                    return 4
                elif mode == "text":
                    return 'high'
                else:
                    raise NotImplementedError()
            elif v<p[5]:
                if mode == "one-hot":
                    return 5
                elif mode == "text":
                    return 'very high'
                else:
                    raise NotImplementedError()
            else:
                if mode == "one-hot":
                    return 6
                elif mode == "text":
                    return 'extremely high'
                else:
                    raise NotImplementedError()
        elif len(p)==4:
            if v<p[0]:
                if mode == "one-hot":
                    return 0
                elif mode == "text":
                    return 'very low'
                else:
                    raise NotImplementedError()
            elif v<p[1]:
                if mode == "one-hot":
                    return 1
                elif mode == "text":
                    return 'low'
                else:
                    raise NotImplementedError()
            elif v<p[2]:
                if mode == "one-hot":
                    return 2
                elif mode == "text":
                    return 'medium'
                else:
                    raise NotImplementedError()
            elif v<p[3]:
                if mode == "one-hot":
                    return 3
                elif mode == "text":
                    return 'high'
                else:
                    raise NotImplementedError()
            else:
                if mode == "one-hot":
                    return 4
                elif mode == "text":
                    return 'very high'
                else:
                    raise NotImplementedError()
        elif len(p)==2:
            if v<p[0]:
                if mode == "one-hot":
                    return 0
                elif mode == "text":
                    return 'low'
                else:
                    raise NotImplementedError()
            elif v<p[1]:
                if mode == "one-hot":
                    return 1
                elif mode == "text":
                    return 'medium'
                else:
                    raise NotImplementedError()
            else:
                if mode == "one-hot":
                    return 2
                elif mode == "text":
                    return 'high'
                else:
                    raise NotImplementedError()
        else:
            raise NotImplementedError()
        
    def scale2value(self, c, p):
        if c == "low" or c == 0:
            v = np.mean([p[0], p[1]])
        elif c == "medium" or c == 1:
            v = np.mean([p[1], p[2]])
        elif c == "high" or c == 2:
            v = np.mean([p[2], p[3]])
        else:
            v = None
        return v
    
    def compare_text(self, x, y):
        
        x = nltk.word_tokenize(x)
        y = nltk.word_tokenize(y)
        
        if x == y:
            return ''
        
        lcs_len, lcs_x_ind, lcs_y_ind = _recon_lcs(x,y)
        
        if lcs_len == 0:
            return '('+ (' '.join(x)) + ') ' + (' '.join(y)) if len(x)>0 or len(y)>0 else ''
        else:
            comparisons =  '\n'.join(['(' + (' '.join(x_i)) + ') ' + (' '.join(y_i)) for x_i, y_i in [(x[:lcs_x_ind], y[:lcs_y_ind]), (x[lcs_x_ind+lcs_len:], y[lcs_y_ind+lcs_len:])] if len(x_i)>0 or len(y_i)>0])
            return comparisons
    
        
    def _load_data(self, business_unit):
        
        spending_column_names = ['Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV']
        funnel_column_names = ['Awareness', 'Consideration', 'Purchase', 'Purchase over Consideration']
        
        def add_auxiliary_columns(dataset_df):
            # add auxiliary columns
            dataset_df['Purchase over Consideration'] = dataset_df['Purchase'].values / dataset_df['Consideration'].values
            for dataset_column_name in list(dataset_df):
                if dataset_column_name.startswith("Text") or dataset_column_name.startswith("Image Caption"):
                    comparisons = list(zip(np.insert(dataset_df[dataset_column_name].values[:-1], 0, ''), dataset_df[dataset_column_name].values))
                    dataset_df[dataset_column_name + ' Diff'] = [self.compare_text(nan2str(prev), nan2str(curr)) for prev, curr in comparisons]
                    dataset_df[dataset_column_name + ' Diff Score'] = [calculate_rouge_l(nan2str(prev), nan2str(curr)) for prev, curr in comparisons]
        
        def calculate_percentiles(dataset_df):
            # calculate percentiles
            spending_percentile_df = dataset_df[spending_column_names].quantile([0] + self.spending_percentiles + [1])
            #spending_percentile_df["Percentile"] = [0] + spending_percentiles + [1]
            funnel_percentile_df = dataset_df[funnel_column_names].quantile([0] + self.funnel_percentiles + [1])
            #funnel_percentile_df["Percentile"] = [0] + funnel_percentiles + [1]
            return spending_percentile_df, funnel_percentile_df
        
        
        filename = 'data/%s_dataset.csv'%business_unit
        dataset_df = pd.read_csv(filename, encoding='utf-8', parse_dates=['Week'])  
        add_auxiliary_columns(dataset_df)
        spending_percentile_df, funnel_percentile_df = calculate_percentiles(dataset_df)
        
        with codecs.open('data/spending_percentiles.csv', 'w', 'utf-8') as csv_file:
            spending_percentile_df.to_csv(csv_file, index=True, lineterminator='\n')
            
        with codecs.open('data/funnel_percentiles.csv', 'w', 'utf-8') as csv_file:
            funnel_percentile_df.to_csv(csv_file, index=True, lineterminator='\n')
            
        spending_percentile_df = spending_percentile_df.iloc[1:-1]
        print(spending_percentile_df)
        funnel_percentile_df = funnel_percentile_df.iloc[1:-1]
        print(funnel_percentile_df)
        
        dataset_column_names = list(dataset_df)
        num_htmls = int(re.search(r'Text (\d+?)\-\d', dataset_column_names[-2]).group(1))
        
        
        docs = []
    
        for idx, row in dataset_df.iterrows():
            
            spendings = []
            # include marketing spending
            for spending_column_name in spending_column_names:
                value = row[spending_column_name] if not is_nan(row[spending_column_name]) else 0
                scale = self.value2scale(value, spending_percentile_df[spending_column_name].values)
                spendings.append(scale)
            
            
            pages = []
            for i in range(num_htmls):
                score_column_name = 'HTML %d Page Views'%(i+1)
                texts = []
                images = []
                texts_n_images = []
                for j in range(4):
                    text_column_name = 'Text %d-%d'%(i+1, j+1)
                    text = ""
                    if text_column_name in dataset_column_names:
                        if self.text_format == "orig":
                            text = nan2str(row[text_column_name])
                        elif self.text_format == "diff":
                            text = nan2str(row[text_column_name + " Diff"])
                        elif self.text_format == "score+diff+orig":
                            text = nan2str(row[text_column_name])
                            diff_text = nan2str(row[text_column_name + " Diff"])
                            diff_text = '\n'.join([x for x in diff_text.split('\n') if not x.startswith('()')])
                            score = row[text_column_name + " Diff Score"]
                            text = ("" if score == 1 else "updates"+(":" if len(diff_text)>0 else ":")+"\n"+ diff_text +("\n" if len(diff_text)>0 else "")) + text
                        else:
                            raise NotImplementedError()
                        texts.append(text)
                        
                    image_column_name = 'Image Caption %d-%d'%(i+1, j+1)
                    image = ""
                    if image_column_name in dataset_column_names:
                        if self.image_format == "orig":
                            image = nan2str(row[image_column_name])
                            image = ("background image: " if len(image)>0 else "") + image
                        elif self.image_format == "score+diff+orig":
                            image = nan2str(row[image_column_name])
                            diff_image = nan2str(row[image_column_name + " Diff"])
                            diff_image = '\n'.join([x for x in diff_image.split('\n') if not x.startswith('()')])
                            score = row[image_column_name + " Diff Score"]
                            image = ("" if score == 1 else "differences"+(":" if len(diff_image)>0 else ":")+"\n"+ diff_image +("\n" if len(diff_image)>0 else "")) + image
                            image = ("background image:\n" if len(image)>0 else "") + image
                        else:
                            raise NotImplementedError()    
                        images.append(image)
                    
                    texts_n_images.append(text + '\n' + image)
                    
                page_score = row[score_column_name] if score_column_name in dataset_column_names and not is_nan(row[score_column_name]) else 0
                pages.append({'text':'\n'.join(texts), 'image': '\n'.join(images), 'text_n_image': '\n'.join(texts_n_images), 'score': page_score})
            
            top_k_idxes = sorted(sorted(range(len(pages)), key = lambda idx: pages[idx]['score'])[-self.k:])
            
            # include top-5 page views
            page_views = []
            page_views.extend([str(pages[idx]['score']) for idx in top_k_idxes])
            
            
            spending_mod = 'marketing spending:\n' + ('\n'.join([spending + ' ' + spending_column_name for spending_column_name, spending in zip(spending_column_names, spendings)]))
            text_n_image_mod = ('\n'.join([pages[idx]['text_n_image'] for idx in top_k_idxes]))
            text_mod = ('\n'.join([pages[idx]['text'] for idx in top_k_idxes]))
            image_mod = ('\n'.join([pages[idx]['image'] for idx in top_k_idxes]))
            if self.modality == "spending+text+image":
                doc = spending_mod + '\n' + text_n_image_mod
            elif self.modality == "spending+text":
                doc = spending_mod + '\n' + text_mod
            elif self.modality == "spending+image":
                doc = spending_mod + '\n' + image_mod
            elif self.modality == "text+image":
                doc = text_n_image_mod
            elif self.modality == "spending":
                doc = spending_mod
            elif self.modality == "text":
                doc = text_mod
            elif self.modality == "image":
                doc = image_mod
            else:
                raise NotImplementedError()
            docs.append(doc)
        labels = []
        label_attrs = []
        for idx, row in dataset_df.iterrows():
            scale = self.value2scale(row['Purchase over Consideration'], funnel_percentile_df['Purchase over Consideration'].values, mode=self.output_type)
            labels.append(scale)
            label_attrs.append((row['Purchase over Consideration'], row["Week"].strftime('%Y-%m-%d')))
        
        
        sns.displot(labels)
        plt.title("%s Labels"%business_unit)
        #plt.show()
        plt.savefig("log/%s_Labels_%s.png"%(business_unit, self.output_type))
        plt.clf()
    
        doc_queue = []
        
        doc_seqs = []
        
        for doc in docs:
            doc_queue.append(doc)
            if len(doc_queue)==self.num_weeks:
                #doc_seqs.append('<pad>' + ('<\s>'.join(doc_queue)))  # "<\s>" - e25 - v0.7 - t0.8
                doc_seqs.append((' \n '.join(doc_queue)))
                doc_queue.pop(0)
                
        doc_lengths = []
        for doc in doc_seqs:
            tokens = nltk.word_tokenize(doc)
            doc_lengths.append(len(tokens))
        doc_lengths = np.array(doc_lengths)
        sns.displot(doc_lengths)
        plt.title("%s Document Lengths"%business_unit)
        #plt.show()
        plt.savefig("log/%s_Document_Lengths_%s.png"%(business_unit, self.modality))
        plt.clf()
        
        labels = labels[self.num_weeks-1:]
        label_attrs = label_attrs[self.num_weeks-1:]
        return doc_seqs, labels, label_attrs
                
    def load_data(self):
        all_doc_seqs = []
        all_labels = []
        all_label_attrs = []
        for business_unit in self.business_units:
            doc_seqs, labels, label_attrs = self._load_data(business_unit)
            all_doc_seqs.extend(doc_seqs)
            all_labels.extend(labels)
            all_label_attrs.extend(label_attrs)
        return all_doc_seqs, all_labels, all_label_attrs
    
    def load_tokenizer(self):
        # Load tokenizer.
        #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
            tokenizer = T5Tokenizer.from_pretrained(self.model_type)
        elif self.model_type.startswith("gpt2"):
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_type)
            tokenizer.pad_token = tokenizer.eos_token
        elif self.model_type.startswith("bert"):
            tokenizer = BertTokenizer.from_pretrained(self.model_type)
        else:
            raise NotImplementedError()
    
        # Add a [CLS] to the vocabulary (we should train it also!)
        #num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        
        #print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
        #print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
        #print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
        #print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))
    
        return tokenizer
    
    
    def init_data_loader(self, dataset):
    
        # Split into training and validation sets
        train_size = round(self.dataset_split[0] * len(dataset))
        val_size = round(self.dataset_split[1] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))
        print('{:>5,} test samples'.format(test_size))
    
    
        # Create the DataLoaders for our training and validation datasets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = self.batch_size # Trains with this batch size.
                )
        
        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                )
        
        test_dataloader = DataLoader(
                    test_dataset, # The validation samples.
                    sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                    batch_size = self.batch_size # Evaluate with this batch size.
                )
        
        return train_dataloader, validation_dataloader, test_dataloader
    
    
    def load_model(self, tokenizer):
        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
            config = T5Config.from_pretrained(self.model_type, output_hidden_states=False)
        elif self.model_type.startswith("gpt2"):
            config = GPT2Config.from_pretrained(self.model_type, output_hidden_states=False)
        elif self.model_type.startswith("bert"):
            config = BertConfig.from_pretrained(self.model_type, output_hidden_states=False)
        else:
            raise NotImplementedError()
        if self.output_type == "one-hot":
            config.num_labels = self.num_labels
        #config.summary_use_proj = True
        #config.summary_proj_to_labels = True
        config.pad_token_id = tokenizer.pad_token_id
        # instantiate the model
        if self.output_type == "one-hot":
            if self.model_type.startswith("gpt2"):
                model = GPT2ForSequenceClassification.from_pretrained(self.model_type, config=config)
            elif self.model_type.startswith("bert"):
                model = BertForSequenceClassification.from_pretrained(self.model_type, config=config)
            else:
                raise NotImplementedError()
        elif self.output_type == "text":
            if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                model = T5ForConditionalGeneration.from_pretrained(self.model_type, config=config)
            elif self.model_type.startswith("gpt2"):
                model = GPT2LMHeadModel.from_pretrained(self.model_type, config=config)
            else:
                raise NotImplementedError()
        model.resize_token_embeddings(len(tokenizer))
    
        return model
    
    def load_model_from_local_path(self, model_path, tokenizer):
        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
            config = T5Config.from_pretrained(model_path, output_hidden_states=False)
        elif self.model_type.startswith("gpt2"):
            config = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
        elif self.model_type.startswith("bert"):
            config = BertConfig.from_pretrained(model_path, output_hidden_states=False)
        else:
            raise NotImplementedError()
        #print("config.num_labels:", config.num_labels)
        #config.summary_use_proj = True
        #config.summary_proj_to_labels = True
        config.pad_token_id = tokenizer.pad_token_id
        # instantiate the model
        if self.output_type == "one-hot":
            if self.model_type.startswith("gpt2"):
                model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
            elif self.model_type.startswith("bert"):
                model = BertForSequenceClassification.from_pretrained(model_path, config=config)
            else:
                raise NotImplementedError()
        elif self.output_type == "text":
            if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
            elif self.model_type.startswith("gpt2"):
                model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
            else:
                raise NotImplementedError()
        model.resize_token_embeddings(len(tokenizer))
    
        return model
        
    
    
    def save_model(self, model, tokenizer):
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        
        output_dir = 'model/model_save_%s_%s_%d/'%(self.model_type, self.modality, self.seed["train"])
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Saving model to %s" % output_dir)
        
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    
    def format_time(self, elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))
        
    
    def train(self, model, epochs, learning_rate, tuning_method, tokenizer, train_dataloader, validation_dataloader):
        # Tell pytorch to run this model on the GPU.
        model.cuda()
        #model.to(device)
        
        if tuning_method == "head-tuning":
            if self.model_type.startswith("gpt2"):
                for name, param in model.transformer.named_parameters():
                    param.requires_grad = False
            elif self.model_type.startswith("bert"):
                for name, param in model.bert.named_parameters():
                    param.requires_grad = False
            else:
                raise NotImplementedError()
    
        
        random.seed(self.seed["train"])
        np.random.seed(self.seed["train"])
        torch.manual_seed(self.seed["train"])
        torch.cuda.manual_seed_all(self.seed["train"])
        
        
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        optimizer = AdamW(model.parameters(),
                          lr = learning_rate,
                          eps = self.epsilon
                        )
        
        
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs
        
        # Create the learning rate scheduler.
        # This changes the learning rate as the training loop progresses
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = self.warmup_steps, 
                                                    num_training_steps = total_steps)
        '''
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1, last_epoch=-1)
        '''
    
        total_t0 = time.time()
        
        training_stats = []
        
        model = model.to(device)
        
        max_val_acc = 0
        
        for epoch_i in range(0, epochs):
        
            # ========================================
            #               Training
            # ========================================
        
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
        
            t0 = time.time()
        
            total_train_loss = 0
        
            model.train()
        
            for step, batch in enumerate(train_dataloader):
        
                b_input_ids = batch[0].to(device)
                b_masks = batch[1].to(device)
                
                b_labels = batch[2]
                if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                    b_label_masks = batch[3].to(device)
        
                model.zero_grad()      
                
                #print("b_labels:", b_labels)  
                
                if self.output_type == "one-hot":
                    outputs = model(  b_input_ids,
                                  attention_mask = b_masks,
                                  token_type_ids=None,
                                  labels = b_labels.to(device)
                                )
                elif self.output_type == "text":
                    if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                        #lm_labels = b_labels.to(device)
                        #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                        outputs = model(
                                    b_input_ids,
                                    attention_mask=b_masks,
                                    #decoder_input_ids=lm_labels,
                                    #decoder_attention_mask=b_label_masks
                                    labels=b_labels.to(device)
                                )
                    elif self.model_type.startswith("gpt2"):
                        outputs = model(  b_input_ids,
                                  attention_mask = b_masks,
                                  token_type_ids=None,
                                  labels = b_input_ids
                                )
                else:
                    raise NotImplementedError()
                
                loss = outputs.loss
        
                batch_loss = loss.item()
                total_train_loss += batch_loss
        
                loss.backward()
        
                optimizer.step()
        
                scheduler.step()
        
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)       
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)
        
            print("")
            print("  Average training loss: {0:.6f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================
        
            print("")
            print("Running Validation...")
        
            t0 = time.time()
        
            model.eval()
        
            total_eval_loss = 0
    
            correct = 0
            
            num_examples = 0
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                b_input_ids = batch[0].to(device)
                b_masks = batch[1].to(device)
                
                b_labels = batch[2]
                if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                    b_label_masks = batch[3].to(device)
                
                with torch.no_grad():        
                    if self.output_type == "one-hot":
                        outputs = model(  b_input_ids,
                                  attention_mask = b_masks,
                                  token_type_ids=None,
                                  labels = b_labels.to(device)
                                )
                    elif self.output_type == "text":
                        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                            #lm_labels = b_labels.to(device)
                            #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                            outputs = model(
                                        b_input_ids,
                                        attention_mask=b_masks,
                                        #decoder_input_ids=lm_labels,
                                        #decoder_attention_mask=b_label_masks
                                        labels=b_labels.to(device)
                                    )
                        elif self.model_type.startswith("gpt2"):
                            outputs = model(  b_input_ids,
                                  attention_mask = b_masks,
                                  token_type_ids=None,
                                  labels = b_input_ids
                                )
                    else:
                        raise NotImplementedError()
        
                    loss = outputs.loss
                    
                    if self.output_type == "one-hot":
                        preds = outputs.logits.argmax(dim=1)
                    elif self.output_type == "text":
                        if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                            generated_ids = model.generate(input_ids=b_input_ids, 
                                  attention_mask=b_masks, 
                                  max_length=b_labels.size(1))
                            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                            b_labels = tokenizer.batch_decode(b_labels.to(device), skip_special_tokens=True)
                        elif self.model_type.startswith("gpt2"):
                            sequence_lengths = torch.ne(b_input_ids, tokenizer.pad_token_id).sum(-1) - len(nltk.word_tokenize(b_labels[0]))
                            preds = []
                            for i, sequence_length in enumerate(sequence_lengths):   
        
                                generated_ids = model.generate(
                                            b_input_ids[i][:sequence_length].unsqueeze(0), 
                                            do_sample=True,   
                                            top_k=50, 
                                            max_length = sequence_length+len(nltk.word_tokenize(b_labels[0])),
                                            top_p=0.95, 
                                            num_return_sequences=1
                                            )[0][-1:]
                                
                                preds.append(tokenizer.decode(generated_ids, skip_special_tokens=False))                       
                            
                    else:
                        raise NotImplementedError()
    
                #print("labels:", b_labels, "predictions:",  preds)
                #correct += preds.eq(b_labels).sum().item()
                correct += sum([l==p for l, p in zip(b_labels, preds)])
                    
                batch_loss = loss.item()
                total_eval_loss += batch_loss       
                
                num_examples += b_input_ids.size(0) 
        
            avg_val_loss = total_eval_loss / len(validation_dataloader)
    
            val_acc = correct / num_examples
            
            validation_time = self.format_time(time.time() - t0)    
        
            print("  Validation Loss: {0:.6f}".format(avg_val_loss))
            print("  Validation Accuracy: {0:.2f}".format(val_acc))
            print("  Validation took: {:}".format(validation_time))
        
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            
            if val_acc >= max_val_acc:
                self.save_model(model, tokenizer)
                max_val_acc = val_acc
        
        print("")
        print("Training complete!")
        print("  Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
        return training_stats
    
    
    
    
    def plot_training_stats(self, data):
        
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=data)
        
        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')
        
        # A hack to force the column headers to wrap.
        #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
        
        # Display the table.
        print(df_stats)
        
        
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12,6)
        
        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
        
        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])
        
        #plt.show()
        plt.savefig("log/Train_Validation_Loss_%s_%s_%d.png"%(self.model_type, self.modality, self.seed["train"]))
        plt.clf()
    
    
    def print_model_params(self, model):
        
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())
        
        print('The model has {:} different named parameters.\n'.format(len(params)))
        
        print('==== Embedding Layer ====\n')
        
        for p in params[0:2]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        
        print('\n==== First Transformer ====\n')
        
        for p in params[2:14]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        
        print('\n==== Output Layer ====\n')
        
        for p in params[-2:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    
    def test(self, model, tokenizer, test_dataloader, **kwargs):
        # ========================================
        #               Test
        # ========================================
    
        print("")
        print("Running Test...")
        model.to(device)
        
        t0 = time.time()
    
        model.eval()
    
        total_test_loss = 0
    
        correct = 0
        
        num_examples = 0
        
        funnel_percentile_df = pd.read_csv('data/funnel_percentiles.csv', index_col=0)
        print(funnel_percentile_df)
        

        
        preds_df = pd.DataFrame({'Week': [],
                           'Label': [],
                       'Label Real': [],
                       'Prediction': [], 
                       'Prediction Approx': []})
        # Evaluate data for one epoch
        for batch in test_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2]
            if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                b_label_masks = batch[3].to(device)
                b_label_attrs = batch[4]
            elif self.model_type.startswith("bert") or self.model_type.startswith("gpt2"):
                b_label_attrs = batch[3]
            else:
                raise NotImplementedError()
                
            with torch.no_grad():     
                
                if self.output_type == "one-hot":
                    outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels.to(device)
                            )
                elif self.output_type == "text":
                    if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                        #lm_labels = b_labels.to(device)
                        #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                        outputs = model(
                                    b_input_ids,
                                    attention_mask=b_masks,
                                    #decoder_input_ids=None,
                                    #decoder_attention_mask=b_label_masks,
                                    labels=b_labels.to(device)
                                )
                    elif self.model_type.startswith("gpt2"):
                        outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_input_ids
                            )
                else:
                    raise NotImplementedError()
    
                loss = outputs.loss
                
                if self.output_type == "one-hot":
                    preds = outputs.logits.argmax(dim=1)
                elif self.output_type == "text":
                    if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                        generated_ids = model.generate(input_ids=b_input_ids, 
                              attention_mask=b_masks, 
                              max_length=b_labels.size(1))
                        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        b_labels = tokenizer.batch_decode(b_labels.to(device), skip_special_tokens=True)
                    elif self.model_type.startswith("gpt2"):
                        sequence_lengths = torch.ne(b_input_ids, tokenizer.pad_token_id).sum(-1) - len(nltk.word_tokenize(b_labels[0]))
                        preds = []
                        for i, sequence_length in enumerate(sequence_lengths):   
    
                            generated_ids = model.generate(
                                        b_input_ids[i][:sequence_length].unsqueeze(0), 
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = sequence_length+len(nltk.word_tokenize(b_labels[0])),
                                        top_p=0.95, 
                                        num_return_sequences=1
                                        )[0][-1:]
                            
                            preds.append(tokenizer.decode(generated_ids, skip_special_tokens=False))         
                else:
                    raise NotImplementedError()   
                
    
            if torch.is_tensor(b_labels):
                b_labels = b_labels.cpu().detach().tolist()
            if torch.is_tensor(preds):
                preds = preds.cpu().detach().tolist()
            for i in range(len(b_label_attrs)):
                if torch.is_tensor(b_label_attrs[i]):
                    b_label_attrs[i] = b_label_attrs[i].cpu().detach().tolist()
    
            #print("labels:", b_labels, "predictions:",  preds)  
            #print("label_attrs:", b_label_attrs)
            b_preds_df = pd.DataFrame({'Week': b_label_attrs[1],
                        'Label': b_labels,
                       'Label Real': b_label_attrs[0],
                       'Prediction': preds, 
                       'Prediction Approx': [self.scale2value(pred, funnel_percentile_df['Purchase over Consideration'].values) for pred in preds]})
            
            preds_df = pd.concat([preds_df,b_preds_df], ignore_index = True)
            #correct += preds.eq(b_labels).sum().item()
            correct += sum([l==p for l, p in zip(b_labels, preds)])
                
            batch_loss = loss.item()
            total_test_loss += batch_loss       
            
            num_examples += b_input_ids.size(0) 
    
        avg_test_loss = total_test_loss / len(test_dataloader)
    
        test_acc = correct / num_examples
        
        test_time = self.format_time(time.time() - t0)    
    
        print(preds_df)
        with codecs.open('log/%s_preds_%s_%s_%d.csv'%(kwargs["segment"] if "segment" in kwargs else "test", self.model_type, self.modality, self.seed["train"]), 'w', 'utf-8') as csv_file:
            preds_df.to_csv(csv_file, index=False, lineterminator='\n')
            
        print("  Test Loss: {0:.6f}".format(avg_test_loss))
        print("  Test Accuracy: {0:.2f}".format(test_acc))
        print("  Test took: {:}".format(test_time))


        results_log = "log/results.xlsx"
        sheet_name = "%s_%s_%s"%(kwargs["segment"] if "segment" in kwargs else "test", self.model_type, self.modality)
        if os.path.exists(results_log):
            xl = pd.ExcelFile(results_log)
            if sheet_name in xl.sheet_names:
                results_df = xl.parse(sheet_name)
            else:
                results_df = pd.DataFrame(["Loss", "Accuracy", "Time"], columns=["Metrics"])
            writer_params = {"path":results_log, "engine":"openpyxl", "mode":"a", "if_sheet_exists":"replace"}    
        else:
            results_df = pd.DataFrame(["Loss", "Accuracy", "Time"], columns=["Metrics"])
            writer_params = {"path":results_log, "engine":"openpyxl", "mode":"w"}
            
        results_df["%d"%self.seed["train"]]=[avg_test_loss, test_acc, test_time]
        with pd.ExcelWriter(**writer_params) as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    
    
    def attribute(self, model, tokenizer, test_dataloader):
        # ========================================
        #               Attribution
        # ========================================
    
        print("")
        print("Running Attribution...")
        model.to(device)
        
        t0 = time.time()
    
        model.eval()
    
        def model_attr_func(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
            return torch.softmax(model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask).logits[:,0], dim=1)
        
        
        ig = T5IntegratedGradients(model_attr_func, model.encoder.embed_tokens)
        #ig = IntegratedGradients(model_attr_func)
        
        
        def summarize_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            return attributions
    
        def normalize_attributions(attributions):
            norm = torch.norm(attributions)
            attributions = attributions / norm
            return attributions
        
        def categorize_attributions(attributions):
            arr_attributions = np.array(attributions.detach().tolist())
            p = [np.percentile(arr_attributions, item) for item in [0.13, 2.27, 15.86, 50, 84.13, 97.72, 99.86]]
            #p = torch.quantile(attributions, torch.tensor([0.13, 2.27, 15.86, 50, 84.13, 97.72, 99.86], dtype = attributions.dtype, device= device))
            scales = torch.zeros_like(attributions)
            scales[attributions < p[0]] = -3
            scales[(attributions >= p[0])&(attributions < p[1])] = -2
            scales[(attributions >= p[1])&(attributions < p[2])] = -1
            scales[(attributions >= p[2])&(attributions < p[4])] = 0
            scales[(attributions >= p[4])&(attributions < p[5])] = 1
            scales[(attributions >= p[5])&(attributions < p[6])] = 2
            scales[attributions >= p[6]] = 3
            return scales
    
        def brute_search(needle, haystack):
            indexes = []
            for i in range(len(haystack)):
                if np.array_equal(haystack[i:len(needle) + i],needle):
                    indexes.append(i)
            return indexes
            
        def get_prompt_mask(input_ids):
            input_ids = input_ids.squeeze().detach().cpu().tolist()
            marketing_spending_ids = tokenizer("marketing spending:")["input_ids"][:-1]
            background_image_ids = tokenizer("background image:")["input_ids"][:-1]
            updates_ids = tokenizer("updates:")["input_ids"][:-1]
            differences_ids = tokenizer("differences:")["input_ids"][:-1]
            
            mask = np.zeros_like(input_ids)
            for prompt_ids in [marketing_spending_ids, background_image_ids, updates_ids, differences_ids]:
                indexes = brute_search(prompt_ids, input_ids)
                for i in indexes:
                    for j in range(len(marketing_spending_ids)):
                        mask[i+j] = 1
            
            return mask.astype(dtype=bool)
        
        
        def wrap(tokens, attributions, categorized_attributions, max_len=20):
            wrapped_content = []
            num_rows = (len(tokens) + max_len - 1)//max_len
            for i in range(num_rows):
                wrapped_content.append((tokens[i*max_len:(i+1)*max_len], attributions[i*max_len:(i+1)*max_len], categorized_attributions[i*max_len:(i+1)*max_len]))
            return wrapped_content
        
        def decode_subwords(token_ids, token_attributions, skip_special_tokens=False):
            space_symbol = b'\xe2\x96\x81'.decode('UTF-8')
            tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
            words = []
            word_attributions = []
            current_word = []
            current_word_attribution = []
            for token, token_attribution in zip(tokens, token_attributions):
                if skip_special_tokens and token in tokenizer.all_special_ids:
                    continue
                if token.startswith(space_symbol) or (token in string.punctuation):
                    if len(current_word) > 0:
                        words.append(tokenizer.convert_tokens_to_string(current_word))
                        word_attributions.append(sum(current_word_attribution))
                        current_word = []
                        current_word_attribution = []
                current_word.append(token)
                current_word_attribution.append(token_attribution)
            if len(current_word) > 0:
                words.append(tokenizer.convert_tokens_to_string(current_word))
                word_attributions.append(sum(current_word_attribution))
    
            return words, torch.tensor(word_attributions, dtype=token_attributions.dtype, device=token_attributions.device)
    
    
    
        def position2token_idx(current_position, token_start_positions):
            token_idx = None
            for idx in range(len(token_start_positions)-1):
                token_start_position = token_start_positions[idx]
                token_end_position = token_start_positions[idx+1]
                if current_position >= token_start_position and current_position < token_end_position:
                    token_idx = idx
                    break
            return token_idx
            
    
    
        importance_html_data = []
        importance_csv_data = []
    
        word_n_attr_excel_data = {"All": [], "Spending": [], "Web Content": [], "Digital": [], "DigitalOOH": [], "OOH": [], "Print": [], "Radio": [], "SEM": [], "SEM 2": [], "Social": [], "TV": [], "Image": [], "Text": []}
        phrase_n_attr_excel_data = {"All": [], "Spending": [], "Web Content": [], "Digital": [], "DigitalOOH": [], "OOH": [], "Print": [], "Radio": [], "SEM": [], "SEM 2": [], "Social": [], "TV": [], "Image": [], "Text": []}
        # Evaluate data for one epoch
        for batch in test_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2]
            if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
                b_label_masks = batch[3].to(device)
            
            batch_size = b_input_ids.size(0)
            
            
            if self.output_type == "one-hot":
                raise NotImplementedError()
            elif self.output_type == "text":
                if self.model_type.startswith("t5") or self.model_type.startswith("google/t5"):
    
                    #decode_start_token_ids = torch.tensor([model.config.decoder_start_token_id]*b_input_ids.size(0)).unsqueeze(1).to(device)
                    target_encodings_dict = tokenizer(['<pad>']*batch_size, truncation=True, max_length=2, padding="max_length")
                    decode_start_token_ids = torch.tensor(target_encodings_dict['input_ids']).to(device)
                    #print("decode_start_token_ids:", decode_start_token_ids)
                    decode_start_attention_mask = torch.tensor(target_encodings_dict['attention_mask']).to(device)
                    #print("decode_start_attention_mask:", decode_start_attention_mask)
                    
    
                    for i in range(batch_size):
                        if self.attr_prompts:
                            attr_mask = ~((b_input_ids[i] == tokenizer.pad_token_id) | (b_input_ids[i] == tokenizer.eos_token_id))
                        else:
                            attr_mask = ~((b_input_ids[i] == tokenizer.pad_token_id) | (b_input_ids[i] == tokenizer.eos_token_id) | torch.tensor(get_prompt_mask(b_input_ids[i]), device=device))
                        #print("attr_mask:", attr_mask.detach().tolist())
                        
                        with torch.no_grad():     
                            scores = model_attr_func(
                                    input_ids=b_input_ids[i].unsqueeze(0),
                                    attention_mask=b_masks[i].unsqueeze(0),
                                    decoder_input_ids=decode_start_token_ids[i].unsqueeze(0),
                                    decoder_attention_mask=decode_start_attention_mask[i].unsqueeze(0)
                                )
                        
                        model.zero_grad()
                        attributions, delta = ig.attribute(inputs=b_input_ids[i].unsqueeze(0),
                                      baseline_method = self.baseline_method, #generate_reference(b_input_ids[i]).unsqueeze(0),
                                      attr_mask = attr_mask, 
                                      tokenizer = tokenizer, 
                                      additional_forward_args = (b_masks[i].unsqueeze(0), decode_start_token_ids[i].unsqueeze(0), decode_start_attention_mask[i].unsqueeze(0)),
                                      target = b_labels[i][0], method = self.attr_method, attribute_to_layer_input = False,
                                      return_convergence_delta = True, internal_batch_size = self.attr_batch_size, n_steps = self.attr_steps, smooth_grad = self.smooth_grad)
                        
                        print("delta:", delta.item())
                        
                        attributions_sum = summarize_attributions(attributions)
                        print("attributions_sum:", attributions_sum.detach().tolist())
                        
                        if self.attr_decode_subwords:
                            masked_tokens, masked_attributions_sum = decode_subwords(b_input_ids[i][attr_mask].detach().tolist(), attributions_sum[attr_mask])
                        else:
                            masked_tokens = tokenizer.convert_ids_to_tokens(b_input_ids[i][attr_mask].detach().tolist())
                            masked_attributions_sum = attributions_sum[attr_mask]
                        
    
                        masked_normalized_attributions_sum = normalize_attributions(masked_attributions_sum)
                        #print("masked_normalized_attributions_sum:", masked_normalized_attributions_sum.detach().tolist())
                        masked_categorized_attributions_sum = categorize_attributions(masked_attributions_sum)
                        #print("masked_categorized_attributions_sum:", masked_categorized_attributions_sum.detach().tolist())
                            
                        # importance html data
                        if self.attr_word_viz == "norm":
                            viz_attributions = masked_normalized_attributions_sum
                        elif self.attr_word_viz == "scale":
                            color_strength = 0.3
                            viz_attributions = masked_categorized_attributions_sum/torch.max(masked_categorized_attributions_sum)*color_strength
                        
                        position_vis = viz.VisualizationDataRecord(
                            viz_attributions,
                            #torch.max(torch.softmax(scores[0], dim=0)),
                            torch.max(scores[0]),
                            tokenizer.decode(torch.argmax(scores), skip_special_tokens=True),
                            tokenizer.decode(b_labels[i], skip_special_tokens=True),
                            tokenizer.decode(b_labels[i], skip_special_tokens=True),
                            attributions_sum.sum(),
                            #np.array(all_tokens)[attr_mask.cpu()],
                            masked_tokens,
                            delta)
                        
                        viz_html = viz.visualize_text([position_vis]).data
                        print("viz_html:", viz_html)
                        importance_html_data.append(viz_html)
                        
                        # importance csv data
                        importance_csv_data.append(['True Label', 'Predicted Label', 'Attribution Label', 'Attribution Score',  'Convergence Score'])
                        importance_csv_data.append([tokenizer.decode(b_labels[i], skip_special_tokens=True), 
                                         tokenizer.decode(torch.argmax(scores), skip_special_tokens=True) + ' (' + str(torch.max(scores[0]).item()) + ')', 
                                         tokenizer.decode(b_labels[i], skip_special_tokens=True), 
                                         attributions_sum.sum().item(), 
                                         delta.item()])
                        importance_csv_data.append(['Word Importance'])
                        for tokens_per_row, attributions_per_row, categorized_attributions_per_row in wrap(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist()):
                            importance_csv_data.append(tokens_per_row)
                            importance_csv_data.append(attributions_per_row)
                            importance_csv_data.append(categorized_attributions_per_row)
                        importance_csv_data.append([])
                        
    
                        # word and attribution score excel data
                        if delta.item() >= self.attr_delta_valid_range[0] and delta.item() < self.attr_delta_valid_range[1] and torch.argmax(scores).item() == b_labels[i][0].item():
                            for token, attribution, categorized_attribution in np.array(list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist()))):
                                word_n_attr_excel_data["All"].append([token, attribution, categorized_attribution])
                            
                            residual = np.ones(shape=(len(masked_tokens),), dtype=bool)
                            
                            # spending
                            token_start_positions = []
                            for token in masked_tokens:
                                token_start_positions.append((0 if len(token_start_positions)==0 else token_start_positions[-1]) + len(token) + 1)
                            if len(token_start_positions) > 0:
                                token_start_positions.insert(0,0)
                            text = " ".join(masked_tokens)
                            spending_match_objs = re.finditer(r"(very )?(low|medium|high) (DigitalOOH|Digital|OOH|Print|Radio|SEM 2|SEM|Social|TV)", text)
                            for match_obj in spending_match_objs:
                                span_start_position = match_obj.start()
                                span_end_position = match_obj.end()
                                phrase = match_obj.group().split(' ')
                                phrase_start_idx = position2token_idx(span_start_position, token_start_positions)
                                phrase_end_idx = position2token_idx(span_end_position+1, token_start_positions)
                                
                                for key_word in ["Digital", "DigitalOOH", "OOH", "Print", "Radio", "2", "SEM",  "Social", "TV"]:
                                    if key_word in phrase:
                                        for token, attribution, categorized_attribution in list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist()))[phrase_start_idx: phrase_end_idx]:
                                            word_n_attr_excel_data["SEM 2" if key_word == "2" else key_word].append([token, attribution, categorized_attribution])
                                        phrase_n_attr_excel_data["SEM 2" if key_word == "2" else key_word].append([' '.join(phrase), attr_phrase_pooling(masked_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]), round(attr_phrase_pooling(masked_categorized_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]))])
                                        break
                                for token, attribution, categorized_attribution in list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist()))[phrase_start_idx: phrase_end_idx]:
                                    word_n_attr_excel_data["Spending"].append([token, attribution, categorized_attribution])
                                phrase_n_attr_excel_data["Spending"].append([' '.join(phrase), attr_phrase_pooling(masked_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]), round(attr_phrase_pooling(masked_categorized_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]))])    
                                residual[phrase_start_idx: phrase_end_idx] = False
                                
                            # web content
                            for token, attribution, categorized_attribution in np.array(list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist())))[residual]:
                                word_n_attr_excel_data["Web Content"].append([token, attribution, categorized_attribution])    
                                
                            # image caption
                            caption_df = pd.read_csv('data/content/processed/iTrade_image_features.csv', usecols = ["Caption"])
                            caption_df = caption_df.drop_duplicates(subset=["Caption"])
                            caption_match_objs = re.finditer("|".join(caption_df["Caption"].values), text)
                            for match_obj in caption_match_objs:
                                span_start_position = match_obj.start()
                                span_end_position = match_obj.end()
                                phrase = match_obj.group().split(' ')
                                phrase_start_idx = position2token_idx(span_start_position, token_start_positions)
                                phrase_end_idx = position2token_idx(span_end_position+1, token_start_positions)
                                for token, attribution, categorized_attribution in list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist()))[phrase_start_idx: phrase_end_idx]:
                                    word_n_attr_excel_data["Image"].append([token, attribution, categorized_attribution])
                                phrase_n_attr_excel_data["Image"].append([' '.join(phrase), attr_phrase_pooling(masked_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]), round(attr_phrase_pooling(masked_categorized_attributions_sum.detach().tolist()[phrase_start_idx: phrase_end_idx]))])    
                                residual[phrase_start_idx: phrase_end_idx] = False
                                
                            # Text
                            for token, attribution, categorized_attribution in np.array(list(zip(masked_tokens, masked_attributions_sum.detach().tolist(), masked_categorized_attributions_sum.detach().tolist())))[residual]:
                                word_n_attr_excel_data["Text"].append([token, attribution, categorized_attribution])    
                        
                        
                elif self.model_type.startswith("gpt2"):
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
    
        importance_html_data = '\n'.join(importance_html_data)
        
        with codecs.open('%s_ig_%s.html'%('word_importance' if self.attr_decode_subwords else 'subword_importance',self.attr_method), 'w', 'utf-8') as file:
            file.write(importance_html_data)
            
        with codecs.open('%s_ig_%s.csv'%('word_importance' if self.attr_decode_subwords else 'subword_importance',self.attr_method), 'w', 'utf-8') as file:
            writer = csv.writer(file) 
            writer.writerows(importance_csv_data)
            
    
        sheet_names = ["All", "Spending", "Web Content", "Digital", "DigitalOOH", "OOH", "Print", "Radio", "SEM", "SEM 2", "Social", "TV", "Image", "Text"]
        writer = pd.ExcelWriter('%s_n_attr.xlsx'%('word' if self.attr_decode_subwords else 'subword'), engine='xlsxwriter')
        for sheet_name in sheet_names:
            df = pd.DataFrame(np.array(word_n_attr_excel_data[sheet_name]), columns=["Word" if self.attr_decode_subwords else "Subword", "Attribution Score", "Attribution Scale"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        
        writer = pd.ExcelWriter('phrase_n_attr.xlsx', engine='xlsxwriter')
        for sheet_name in sheet_names:
            if len(phrase_n_attr_excel_data[sheet_name]) > 0:
                df = pd.DataFrame(np.array(phrase_n_attr_excel_data[sheet_name]), columns=["Phrase", "Attribution Score", "Attribution Scale"])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        
        attribute_time = self.format_time(time.time() - t0)    
    
        print("  Attribution took: {:}".format(attribute_time))
        
    def plot_attribution_stats(self): 
        def token_n_scale2scale2token(tokens, scales):
            scale2token = {}
            for token, scale in zip(tokens, scales):
                #if scale != 0:
                if scale not in scale2token:
                    scale2token[scale] = OrderedDict()
                if token not in scale2token[scale]:
                    scale2token[scale][token] = 0
                scale2token[scale][token] += 1
            return scale2token
    
        def scale2token2df(scale2token):
            dfs = []
            for scale in sorted(scale2token):
                token_n_count = scale2token[scale]
                token_n_count = np.array(list(token_n_count.items())).transpose()
                pd_data = {}
                pd_data[scale] = token_n_count[0]
                pd_data[str(int(scale)) + " Count"] = token_n_count[1]
                dfs.append(pd.DataFrame(pd_data))
            df = pd.concat(dfs, ignore_index=False, axis=1)
            return df
    
    
        def floored_percentage(val, digits):
            val *= 10 ** (digits + 2)
            return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)
        
        
        sheet_names = ["All", "Spending", "Web Content", "Digital", "DigitalOOH", "OOH", "Print", "Radio", "SEM", "SEM 2", "Social", "TV", "Image", "Text"]   
        writer = pd.ExcelWriter('%s_hist.xlsx'%('word' if self.attr_decode_subwords else 'subword'), engine='xlsxwriter')
        for sheet_name in sheet_names:
            df = pd.read_excel('%s_n_attr.xlsx'%('word' if self.attr_decode_subwords else 'subword'), sheet_name=sheet_name, usecols=["Word" if self.attr_decode_subwords else "Subword", "Attribution Scale"])
            df = scale2token2df(token_n_scale2scale2token(df["Word" if self.attr_decode_subwords else "Subword"].values, df["Attribution Scale"].values))
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
        
        writer = pd.ExcelWriter('%s_area.xlsx'%('word' if self.attr_decode_subwords else 'subword'), engine='xlsxwriter')
        score_dfs = {}
        maxs = {}
        mins = {}
        for sheet_name in sheet_names:
            score_dfs[sheet_name] = pd.read_excel('%s_n_attr.xlsx'%('word' if self.attr_decode_subwords else 'subword'), sheet_name=sheet_name, usecols=["Word" if self.attr_decode_subwords else "Subword", "Attribution Score"])
            maxs[sheet_name] = score_dfs[sheet_name]["Attribution Score"].max()
            mins[sheet_name] = score_dfs[sheet_name]["Attribution Score"].min()
        max_abs = np.max(np.concatenate((np.abs(list(maxs.values())), np.abs(list(mins.values())))))
        pct_ranges = [(-1, -0.25)] + [ ((i-5)*5/100, (i-4)*5/100) for i in range(10)] + [(0.25, 1)]
        score_ranges = [tuple(max_abs*np.array(pct_range)) for pct_range in pct_ranges]
        area_df = pd.concat([(score_dfs[sheet_name]["Attribution Score"].groupby(pd.cut(score_dfs[sheet_name]["Attribution Score"],[score_ranges[0][0]-self.epsilon]+[score_range[1] for score_range in score_ranges])).count()).rename(sheet_name) for sheet_name in sheet_names], axis=1).reset_index(level=0)
        area_df.insert(0,"Percentage",["(%s, %s]"%(floored_percentage(pct_range[0],0),floored_percentage(pct_range[1],0)) for pct_range in pct_ranges])
        area_df.to_excel(writer, sheet_name="Area", index=False)
        writer.close()
        
        writer = pd.ExcelWriter('phrase_hist.xlsx', engine='xlsxwriter')
        for sheet_name in sheet_names[3:12]:
            df = pd.read_excel('phrase_n_attr.xlsx', sheet_name=sheet_name, usecols=["Phrase", "Attribution Scale"])
            df = scale2token2df(token_n_scale2scale2token(df["Phrase"].values, df["Attribution Scale"].values))
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
    
        def plot_hist_n_est(sheet_names, col_name, palette, size, chart="hist+kde", element="step", inset=False, ins_xlim=None, ins_ylim=None, ins_zoom=None, ins_loc=None, ins_mark_loc=(1,3)):
            if chart == "kde":
                title = "Kernel Density Estimation"
                filename = title.replace(' ', '_')
            elif chart == "hist+kde":
                title = "Histogram and Kernel Density Estimation"
                filename = title.replace(' ', '_')
            elif chart == "cuhist+kde":
                title = "Cumulative Histogram and Kernel Density Estimation"
                filename = title.replace(' ', '_')
            elif chart == "cuhist+density+kde":
                title = "Normalized Cumulative Histogram and Kernel Density Estimation"
                filename = "Normalized Cumulative Histogram (Density) and Kernel Density Estimation".replace(' ', '_')
            elif chart == "cuhist+probability+kde":
                title = "Normalized Cumulative Histogram and Kernel Density Estimation"
                filename = "Normalized Cumulative Histogram (Probability) and Kernel Density Estimation".replace(' ', '_')
            elif chart == "cdf":
                title = "Cumulative Distribution Function"
                filename = title.replace(' ', '_')
            else:
                raise NotImplementedError()
            
            dfs = []
            for sheet_name in sheet_names:
                df = pd.read_excel('%s_n_attr.xlsx'%('word' if self.attr_decode_subwords else 'subword'), sheet_name=sheet_name, usecols=[col_name])
                dfs.append(df)
            fig, ax = plt.subplots(figsize=size)
            if chart == "kde":
                sns.kdeplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
            elif chart == "hist+kde":
                sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)))
            elif chart == "cuhist+kde":
                sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True)
            elif chart == "cuhist+density+kde":
                sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="density", common_norm=False)
            elif chart == "cuhist+probability+kde":
                sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="probability", common_norm=False)
            elif chart == "cdf":
                sns.ecdfplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
            else:
                raise NotImplementedError()
                
            ax.set_title("%s %s"%(col_name,title))
            ax.set_xlabel(col_name)
            #ax.set_ylabel("Count")
            legend = ax.get_legend()
            handles = legend.legendHandles
            legend.remove()
            ax.legend(handles, sheet_names)
            if inset:
                axins = zoomed_inset_axes(ax, ins_zoom, loc=ins_loc)
                if chart == "kde":
                    sns.kdeplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
                elif chart == "hist+kde":
                    sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)))
                elif chart == "cuhist+kde":
                    sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True)
                elif chart == "cuhist+density+kde":
                    sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="density", common_norm=False)
                elif chart == "cuhist+probability+kde":
                    sns.histplot(pd.concat(dfs, ignore_index=True, axis=1), element=element, fill=False, kde=True, palette=sns.color_palette(palette, len(palette)), cumulative=True, stat="probability", common_norm=False)
                elif chart == "cdf":
                    sns.ecdfplot(pd.concat(dfs, ignore_index=True, axis=1), palette=sns.color_palette(palette, len(palette)))
                else:
                    raise NotImplementedError()
                axins.set_xlim(*ins_xlim)
                axins.set_ylim(*ins_ylim)
                #plt.xticks(visible=False)
                plt.yticks(visible=False)
                legend = axins.get_legend()
                handles = legend.legendHandles
                legend.remove()
                axins.set(xlabel=None)
                axins.set(ylabel=None)
                mark_inset(ax, axins, loc1=ins_mark_loc[0], loc2=ins_mark_loc[1], fc="none", ec="0.5")
            plt.draw()
            #plt.show()
            plt.savefig("%s_%s_%s.png"%(col_name.replace(' ', '_'), filename, ('_'.join(sheet_names)).replace(' ', '_')))
            plt.clf()
            plt.close() 
        plot_hist_n_est(sheet_names[1:3], col_name="Attribution Score" ,palette=["#009DD6", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="kde", inset=False)
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(0.025, 0.125), ins_ylim=(0, 2), ins_zoom=3, ins_loc="center right", ins_mark_loc=(2,4))
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+kde", inset=False)
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+density+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cuhist+probability+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")
        plot_hist_n_est(sheet_names[3:12], col_name="Attribution Score", palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], size=(12,9), chart="cdf", inset=False)
        plot_hist_n_est(sheet_names[12:], col_name="Attribution Score", palette=["#333333", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
        plot_hist_n_est(sheet_names[1:2] + sheet_names[12:], col_name="Attribution Score", palette=["#009DD6", "#333333", "#EC111A"], size=(12,9), chart="hist+kde", inset=True, ins_xlim=(-0.05, 0.05), ins_ylim=(0, 50), ins_zoom=5, ins_loc="center left")
        plot_hist_n_est(sheet_names[1:2] + sheet_names[12:], col_name="Attribution Score", palette=["#009DD6", "#333333", "#EC111A"], size=(12,9), chart="cuhist+density+kde", inset=True, ins_xlim=(0, 0.1), ins_ylim=(0.8, 1), ins_zoom=3, ins_loc="center right")
        
        def plot_cat(sheet_names, col_name, palette=["#333333", "#EC111A","#9E480E","#C0C0C0", "#0C8B72", "#009DD6", "#264478", "#7849B8", "#FF949A"], chart="phrase-violin"):
            span_type = chart.split('-')[0]
            cat_types = chart.split('-')[1].split('+')
            dfs = []
            for sheet_name in sheet_names:
                df = pd.read_excel('%s_n_attr.xlsx'%span_type, sheet_name=sheet_name, usecols=[col_name])
                df = df.rename(columns={col_name:sheet_name})
                dfs.append(df)
            df = pd.concat(dfs, ignore_index=False, axis=1)
            df = pd.melt(df, var_name="Channel", value_name=col_name)
            
            for cat_type in cat_types:
                if cat_type == 'swarm':
                    sns.swarmplot(data=df, x=col_name, y="Channel", size=2, palette=sns.color_palette(["#000000"], len(["#000000"])))   
                elif cat_type == 'violin':
                    rel=sns.catplot(data=df, x=col_name, y="Channel", kind=cat_type, palette=sns.color_palette(palette, len(palette)), color=".9", inner=None, height=9, aspect=1)
                else:
                    rel=sns.catplot(data=df, x=col_name, y="Channel", kind=cat_type, palette=sns.color_palette(palette, len(palette)), height=9, aspect=1)
                    
            rel.fig.suptitle("%s %s %s Chart"%(span_type.title(), col_name, ' and '.join([cat_type.title() for cat_type in cat_types])))
            
            plt.savefig("%s_%s_%s_%s.png"%(span_type.title(), col_name.replace(' ', '_'), '_'.join([cat_type.title() for cat_type in cat_types]), ('_'.join(sheet_names[3:12])).replace(' ', '_')))
            plt.clf()
            plt.close()
        
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-violin")
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-violin"%('word' if self.attr_decode_subwords else 'subword'))
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-violin")
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-violin"%('word' if self.attr_decode_subwords else 'subword'))
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-violin+swarm")
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-violin+swarm"%('word' if self.attr_decode_subwords else 'subword'))
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-violin+swarm")
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-violin+swarm"%('word' if self.attr_decode_subwords else 'subword'))
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="phrase-boxen")
        plot_cat(sheet_names[3:12], col_name="Attribution Scale", chart="%s-boxen"%('word' if self.attr_decode_subwords else 'subword'))
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="phrase-boxen")
        plot_cat(sheet_names[3:12], col_name="Attribution Score", chart="%s-boxen"%('word' if self.attr_decode_subwords else 'subword'))
    
    
    def preprocess(self, tokenizer):
        # load data
        random.seed(self.seed["preprocess"])
        np.random.seed(self.seed["preprocess"])
        torch.manual_seed(self.seed["preprocess"])
        torch.cuda.manual_seed_all(self.seed["preprocess"])
        
        doc_seqs, labels, label_attrs = self.load_data()
        dataset = Dataset(doc_seqs, labels, label_attrs, tokenizer, self.model_type, self.output_type, max_length=self.seq_len)
        train_dataloader, validation_dataloader, test_dataloader = self.init_data_loader(dataset)
        return train_dataloader, validation_dataloader, test_dataloader
    
    def __call__(self, mode="train+test+attribute"):
        
        # load tokenizer
        tokenizer = self.load_tokenizer()
        train_dataloader, validation_dataloader, test_dataloader = self.preprocess(tokenizer)
        
        if "train" in mode:
            model = self.load_model(tokenizer)
            self.print_model_params(model)
            
            training_stats = self.train(model, self.epochs, self.learning_rate, "fine-tuning", tokenizer, train_dataloader, validation_dataloader)
            self.plot_training_stats(training_stats)
            
        if "test" in mode:
            model = self.load_model_from_local_path('model/model_save_%s_%s_%d/'%(self.model_type, self.modality, self.seed["train"]), tokenizer)
            self.test(model, tokenizer, test_dataloader)
            self.test(model, tokenizer, train_dataloader, segment="train")
            self.test(model, tokenizer, validation_dataloader, segment="validation")
            
        if "attribute" in mode:
            model = self.load_model_from_local_path('model/model_save_%s_%s_%d/'%(self.model_type, self.modality, self.seed["train"]), tokenizer)
            self.attribute(model, tokenizer, test_dataloader)
    
            self.plot_attribution_stats()

'''
def consolidate():
    results_log = "log/results.xlsx"
    xl = pd.ExcelFile(results_log)
    agg_df = pd.DataFrame(["Mean", "Std", "Var"], columns=["Metrics"])   
    for sheet_name in xl.sheet_names:
        if sheet_name != "summary":
            sheet_df = xl.parse(sheet_name)
            acc_df = sheet_df.loc[sheet_df['Metrics'] == "Accuracy"]
            acc_df = acc_df.drop("Metrics", axis=1)
            acc_mean = acc_df.mean(axis=1).tolist()[0]
            acc_std = acc_df.std(axis=1).tolist()[0]
            acc_var = acc_df.var(axis=1).tolist()[0]
            agg_df[sheet_name] = [acc_mean, acc_std, acc_var]
    
    writer_params = {"path":results_log, "engine":"openpyxl", "mode":"a", "if_sheet_exists":"replace"}    
    with pd.ExcelWriter(**writer_params) as writer:
        agg_df.to_excel(writer, sheet_name="summary", index=False)
'''
def consolidate():
    
    xl = pd.ExcelFile("log/results.xlsx")
    sheet_dfs = []
    for sheet_name in xl.sheet_names:
        try:
            sheet_df = xl.parse(sheet_name)
            acc_df = sheet_df.loc[sheet_df['Metrics'] == "Accuracy"]
            acc_df = acc_df.drop("Metrics", axis=1)
            dataset, model, modalities = sheet_name.split('_')
            num_rows = len(sheet_df.index)
            sheet_df["Dataset"] = [dataset]*num_rows
            sheet_df["Model"] = [model]*num_rows
            sheet_df["Modalities"] = [modalities]*num_rows
            sheet_df["Mean"] = [None, acc_df.mean(axis=1).tolist()[0], None]
            sheet_df["Std"] = [None, acc_df.std(axis=1).tolist()[0], None]
            sheet_df["Var"] = [None, acc_df.var(axis=1).tolist()[0], None]
            sheet_dfs.append(sheet_df)
        except:
            print('An error occurred while parsing the "%s" sheet.'%sheet_name)
    summary_df = pd.concat(sheet_dfs, axis=0, ignore_index=True) 
    writer_params = {"path":"log/results_summary.xlsx", "engine":"openpyxl", "mode":"w"}    
    with pd.ExcelWriter(**writer_params) as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        
    # plot
    xl = pd.ExcelFile("log/results_summary.xlsx")
    summary_df = xl.parse(xl.sheet_names[0])
    acc_df = summary_df.loc[summary_df["Metrics"] == "Accuracy"]
    for dataset in acc_df["Dataset"].unique().tolist():
        sub_acc_df = acc_df.loc[acc_df["Dataset"] == dataset]
        sub_acc_df = sub_acc_df.drop(["Mean", "Std", "Var", "Metrics", "Dataset"], axis=1)
        sub_acc_df = pd.melt(sub_acc_df, id_vars=["Model", "Modalities"], 
                      var_name="Seed", value_name="Accuracy")
        fig, ax = plt.subplots(figsize=(12,9))
        sns.boxenplot(data=sub_acc_df, x="Accuracy", y="Modalities", hue=sub_acc_df["Model"], gap=.2)
        y_labels = [textwrap.fill(y_label.get_text().replace('+', ' & '), 9) for y_label in ax.get_yticklabels()]
        ax.set_yticklabels(y_labels)
        plt.savefig("log/%s_results_summary.png"%dataset)
        plt.clf()
        plt.close()
            
def main():
    
    experiments = []
    modality_combs = ["spending+text+image", "spending+text", "spending+image", "text+image", "spending", "text", "image"]
    models = [{"model_type": "bert-base-uncased", "output_type": "one-hot"}, {"model_type": "gpt2", "output_type": "one-hot"}, {"model_type": "t5-base", "output_type": "text"}]
    for model in models:
        for modality_comb in modality_combs:
            for i in range(10):
                experiments.append(Experiment(seed={"preprocess": 4, "train": i}, modality=modality_comb, model_type=model["model_type"], output_type=model["output_type"], epochs=5))
            
    for experiment in experiments:
        experiment(mode="train+test")
        
    consolidate()
        
            
if __name__ == '__main__':
    main()
    
