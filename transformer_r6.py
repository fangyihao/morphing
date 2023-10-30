from random import randint

import codecs
import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch_geometric.datasets import coma

torch.manual_seed(4)

from transformers import GPT2Tokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2PreTrainedModel, GPT2ForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import nltk
nltk.download('punkt')

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
from typing import Any, Callable, List, Tuple, Union, overload

from captum._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_outputs,
)
from captum._utils.gradient import _forward_layer_eval, _run_forward
from captum._utils.typing import BaselineType, Literal, ModuleOrModuleList, TargetType
from captum.attr._utils.common import (
    _format_input_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn.parallel.scatter_gather import scatter
import torchvision


#hyper-parameters
business_units = ['iTrade']
modality = "spending+text+image"
output_type = "text" # "one-hot"
model_type = "t5-base" # "t5-base" "bert-base-uncased" "gpt2" "google/t5-v1_1-base"
text_format = "score+diff+orig"  # "orig" "diff" "score+diff+orig"
image_format = "score+diff+orig" # "orig" "score+diff+orig"
k = 5  # top-k pages
batch_size = 8
num_labels = 3   # number of classes to predict 0: conversion value is not volatile 1: conversion value is volatile and increasing 2: conversion value is volatile and decreasing
epochs = 25
learning_rate = 1e-4
warmup_steps = 1e2
epsilon = 1e-8
num_weeks = 2
seq_len = 1280
seed_val = 4
dataset_split = [0.7, 0.10, 0.20]
attr_method = 'gausslegendre' #'riemann_right'
baseline_method = 'gaussblur' # 'pad'

device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device("cuda")



def is_nan(x):
    return (x != x)

def nan2str(x):
    return x if not is_nan(x) else ""

def value2scale(v, p):
    if len(p)==4:
        if v<p[0]:
            return 'very low'
        elif v<p[1]:
            return 'low'
        elif v<p[2]:
            return 'medium'
        elif v<p[3]:
            return 'high'
        else:
            return 'very high'
    elif len(p)==2:
        if v<p[0]:
            return 'low'
        elif v<p[1]:
            return 'medium'
        else:
            return 'high'
    else:
        raise NotImplementedError()

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


def compare_text(x, y):
    
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
    
    

    
def _load_data(business_unit):
    
    spending_column_names = ['Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV']
    funnel_column_names = ['Awareness', 'Consideration', 'Purchase', 'Purchase over Consideration']
    
    def add_auxiliary_columns(dataset_df):
        # add auxiliary columns
        dataset_df['Purchase over Consideration'] = dataset_df['Purchase'].values / dataset_df['Consideration'].values
        for dataset_column_name in list(dataset_df):
            if dataset_column_name.startswith("Text") or dataset_column_name.startswith("Image Caption"):
                comparisons = list(zip(np.insert(dataset_df[dataset_column_name].values[:-1], 0, ''), dataset_df[dataset_column_name].values))
                dataset_df[dataset_column_name + ' Diff'] = [compare_text(nan2str(prev), nan2str(curr)) for prev, curr in comparisons]
                dataset_df[dataset_column_name + ' Diff Score'] = [calculate_rouge_l(nan2str(prev), nan2str(curr)) for prev, curr in comparisons]
    
    def calculate_percentiles(dataset_df):
        # calculate percentiles
        spending_percentile_df = dataset_df[spending_column_names].quantile([0.2,0.4,0.6,0.8])
        funnel_percentile_df = dataset_df[funnel_column_names].quantile([0.3333,0.6667])
        return spending_percentile_df, funnel_percentile_df
    
    
    filename = 'data/%s_dataset.csv'%business_unit
    dataset_df = pd.read_csv(filename, encoding='utf-8', parse_dates=['Week'])  
    add_auxiliary_columns(dataset_df)
    spending_percentile_df, funnel_percentile_df = calculate_percentiles(dataset_df)
    
    dataset_column_names = list(dataset_df)
    num_htmls = int(re.search(r'Text (\d+?)\-\d', dataset_column_names[-2]).group(1))
    
    
    docs = []

    for idx, row in dataset_df.iterrows():
        
        spendings = []
        # include marketing spending
        for spending_column_name in spending_column_names:
            value = row[spending_column_name] if not is_nan(row[spending_column_name]) else 0
            scale = value2scale(value, spending_percentile_df[spending_column_name].values)
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
                    if text_format == "orig":
                        text = nan2str(row[text_column_name])
                    elif text_format == "diff":
                        text = nan2str(row[text_column_name + " Diff"])
                    elif text_format == "score+diff+orig":
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
                    if image_format == "orig":
                        image = nan2str(row[image_column_name])
                        image = ("background image: " if len(image)>0 else "") + image
                    elif image_format == "score+diff+orig":
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
        
        top_k_idxes = sorted(sorted(range(len(pages)), key = lambda idx: pages[idx]['score'])[-k:])
        
        # include top-5 page views
        page_views = []
        page_views.extend([str(pages[idx]['score']) for idx in top_k_idxes])
        
        
        spending_mod = 'marketing spending:\n' + ('\n'.join([spending + ' ' + spending_column_name for spending_column_name, spending in zip(spending_column_names, spendings)]))
        text_n_image_mod = ('\n'.join([pages[idx]['text_n_image'] for idx in top_k_idxes]))
        text_mod = ('\n'.join([pages[idx]['text'] for idx in top_k_idxes]))
        image_mod = ('\n'.join([pages[idx]['image'] for idx in top_k_idxes]))
        if modality == "spending+text+image":
            doc = spending_mod + '\n' + text_n_image_mod
        elif modality == "spending+text":
            doc = spending_mod + '\n' + text_mod
        elif modality == "spending+image":
            doc = spending_mod + '\n' + image_mod
        elif modality == "text+image":
            doc = text_n_image_mod
        elif modality == "spending":
            doc = spending_mod
        elif modality == "text":
            doc = text_mod
        elif modality == "image":
            doc = image_mod
        else:
            raise NotImplementedError()
        docs.append(doc)
    labels = []
    for idx, row in dataset_df.iterrows():
        scale = value2scale(row['Purchase over Consideration'], funnel_percentile_df['Purchase over Consideration'].values)
        labels.append(scale)
    
    
    sns.displot(labels)
    plt.title("%s Labels"%business_unit)
    plt.show()

    doc_queue = []
    
    doc_seqs = []
    
    for doc in docs:
        doc_queue.append(doc)
        if len(doc_queue)==num_weeks:
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
    plt.show()
    
    labels = labels[num_weeks-1:]
    
    return doc_seqs, labels
            
def load_data():
    all_doc_seqs = []
    all_labels = []
    for business_unit in business_units:
        doc_seqs, labels = _load_data(business_unit)
        all_doc_seqs.extend(doc_seqs)
        all_labels.extend(labels)
    return all_doc_seqs, all_labels

def load_tokenizer():
    # Load tokenizer.
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    if model_type.startswith("t5") or model_type.startswith("google/t5"):
        tokenizer = T5Tokenizer.from_pretrained(model_type)
    elif model_type.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type.startswith("bert"):
        tokenizer = BertTokenizer.from_pretrained(model_type)
    else:
        raise NotImplementedError()

    # Add a [CLS] to the vocabulary (we should train it also!)
    #num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    
    #print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
    #print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
    #print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
    #print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

    return tokenizer



class Dataset(Dataset):

    def __init__(self, source, target, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.source = source
        self.source_ids = []
        self.source_attn_masks = []
        self.target = target
        self.target_ids = []
        self.target_attn_masks = []
        #self.cls_token_locations = []
        target_max_length = max([len(nltk.word_tokenize(t_i)) for t_i in target]) +1
    
        for s_i, t_i in zip(source, target):
            if output_type == "one-hot":
                if model_type.startswith("gpt2"):
                    source_encodings_dict = tokenizer('<|startoftext|>'+ s_i + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
                elif model_type.startswith("bert"):
                    source_encodings_dict = tokenizer(s_i, truncation=True, padding="max_length")
                else:
                    raise NotImplementedError()
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    source_encodings_dict = tokenizer(s_i, truncation=True, max_length=max_length, padding="max_length")
                    target_encodings_dict = tokenizer(t_i, truncation=True, max_length=target_max_length, padding="max_length")
                elif model_type.startswith("gpt2"):
                    source_encodings_dict = tokenizer('<|startoftext|>'+ s_i + '\n' + t_i + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
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
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            return self.source_ids[idx], self.source_attn_masks[idx], self.target_ids[idx], self.target_attn_masks[idx]
        else:
            return self.source_ids[idx], self.source_attn_masks[idx], self.target[idx]
        


def init_data_loader(dataset):

    # Split into training and validation sets
    train_size = round(dataset_split[0] * len(dataset))
    val_size = round(dataset_split[1] * len(dataset))
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
                batch_size = batch_size # Trains with this batch size.
            )
    
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    return train_dataloader, validation_dataloader, test_dataloader






def load_model(tokenizer):
    if model_type.startswith("t5") or model_type.startswith("google/t5"):
        config = T5Config.from_pretrained(model_type, output_hidden_states=False)
    elif model_type.startswith("gpt2"):
        config = GPT2Config.from_pretrained(model_type, output_hidden_states=False)
    elif model_type.startswith("bert"):
        config = BertConfig.from_pretrained(model_type, output_hidden_states=False)
    else:
        raise NotImplementedError()
    if output_type == "one-hot":
        config.num_labels = num_labels
    #config.summary_use_proj = True
    #config.summary_proj_to_labels = True
    config.pad_token_id = tokenizer.pad_token_id
    # instantiate the model
    if output_type == "one-hot":
        if model_type.startswith("gpt2"):
            model = GPT2ForSequenceClassification.from_pretrained(model_type, config=config)
        elif model_type.startswith("bert"):
            model = BertForSequenceClassification.from_pretrained(model_type, config=config)
        else:
            raise NotImplementedError()
    elif output_type == "text":
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            model = T5ForConditionalGeneration.from_pretrained(model_type, config=config)
        elif model_type.startswith("gpt2"):
            model = GPT2LMHeadModel.from_pretrained(model_type, config=config)
        else:
            raise NotImplementedError()
    model.resize_token_embeddings(len(tokenizer))

    return model

def load_model_from_local_path(model_path, tokenizer):
    if model_type.startswith("t5") or model_type.startswith("google/t5"):
        config = T5Config.from_pretrained(model_path, output_hidden_states=False)
    elif model_type.startswith("gpt2"):
        config = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
    elif model_type.startswith("bert"):
        config = BertConfig.from_pretrained(model_path, output_hidden_states=False)
    else:
        raise NotImplementedError()
    #print("config.num_labels:", config.num_labels)
    #config.summary_use_proj = True
    #config.summary_proj_to_labels = True
    config.pad_token_id = tokenizer.pad_token_id
    # instantiate the model
    if output_type == "one-hot":
        if model_type.startswith("gpt2"):
            model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
        elif model_type.startswith("bert"):
            model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        else:
            raise NotImplementedError()
    elif output_type == "text":
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
        elif model_type.startswith("gpt2"):
            model = GPT2LMHeadModel.from_pretrained(model_path, config=config)
        else:
            raise NotImplementedError()
    model.resize_token_embeddings(len(tokenizer))

    return model
    



def save_model():    
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    
    output_dir = './model_save/'
    
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

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))
    

def train(model, epochs, learning_rate, tuning_method, tokenizer):
    # Tell pytorch to run this model on the GPU.
    model.to(device)
    if tuning_method == "head-tuning":
        if model_type.startswith("gpt2"):
            for name, param in model.transformer.named_parameters():
                param.requires_grad = False
        elif model_type.startswith("bert"):
            for name, param in model.bert.named_parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()

    
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    optimizer = AdamW(model.parameters(),
                      lr = learning_rate,
                      eps = epsilon
                    )
    
    
    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = warmup_steps, 
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
            if model_type.startswith("t5") or model_type.startswith("google/t5"):
                b_label_masks = batch[3].to(device)
    
            model.zero_grad()        
            
            if output_type == "one-hot":
                outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels.to(device)
                            )
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    #lm_labels = b_labels.to(device)
                    #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_masks,
                                #decoder_input_ids=lm_labels,
                                #decoder_attention_mask=b_label_masks
                                labels=b_labels.to(device)
                            )
                elif model_type.startswith("gpt2"):
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
        training_time = format_time(time.time() - t0)
    
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
            if model_type.startswith("t5") or model_type.startswith("google/t5"):
                b_label_masks = batch[3].to(device)
            
            with torch.no_grad():        
                if output_type == "one-hot":
                    outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels.to(device)
                            )
                elif output_type == "text":
                    if model_type.startswith("t5") or model_type.startswith("google/t5"):
                        #lm_labels = b_labels.to(device)
                        #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                        outputs = model(
                                    b_input_ids,
                                    attention_mask=b_masks,
                                    #decoder_input_ids=lm_labels,
                                    #decoder_attention_mask=b_label_masks
                                    labels=b_labels.to(device)
                                )
                    elif model_type.startswith("gpt2"):
                        outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_input_ids
                            )
                else:
                    raise NotImplementedError()
    
                loss = outputs.loss
                
                if output_type == "one-hot":
                    preds = outputs.logits.argmax(dim=1)
                elif output_type == "text":
                    if model_type.startswith("t5") or model_type.startswith("google/t5"):
                        generated_ids = model.generate(input_ids=b_input_ids, 
                              attention_mask=b_masks, 
                              max_length=b_labels.size(1))
                        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        b_labels = tokenizer.batch_decode(b_labels.to(device), skip_special_tokens=True)
                    elif model_type.startswith("gpt2"):
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

            print("labels:", b_labels, "predictions:",  preds)
            #correct += preds.eq(b_labels).sum().item()
            correct += sum([l==p for l, p in zip(b_labels, preds)])
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss       
            
            num_examples += b_input_ids.size(0) 
    
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        val_acc = correct / num_examples
        
        validation_time = format_time(time.time() - t0)    
    
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
            save_model()
            max_val_acc = val_acc
    
    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    return training_stats




def plot_training_stats(data):
    
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
    
    plt.show()



def print_model_params():
    
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


def test(model, tokenizer):
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
    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2]
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            b_label_masks = batch[3].to(device)
        
        with torch.no_grad():     
            
            if output_type == "one-hot":
                outputs = model(  b_input_ids,
                          attention_mask = b_masks,
                          token_type_ids=None,
                          labels = b_labels.to(device)
                        )
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    #lm_labels = b_labels.to(device)
                    #lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_masks,
                                #decoder_input_ids=None,
                                #decoder_attention_mask=b_label_masks,
                                labels=b_labels.to(device)
                            )
                elif model_type.startswith("gpt2"):
                    outputs = model(  b_input_ids,
                          attention_mask = b_masks,
                          token_type_ids=None,
                          labels = b_input_ids
                        )
            else:
                raise NotImplementedError()

            loss = outputs.loss
            
            if output_type == "one-hot":
                preds = outputs.logits.argmax(dim=1)
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    generated_ids = model.generate(input_ids=b_input_ids, 
                          attention_mask=b_masks, 
                          max_length=b_labels.size(1))
                    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    b_labels = tokenizer.batch_decode(b_labels.to(device), skip_special_tokens=True)
                elif model_type.startswith("gpt2"):
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
            

        print("labels:", b_labels, "predictions:",  preds)    
        #correct += preds.eq(b_labels).sum().item()
        correct += sum([l==p for l, p in zip(b_labels, preds)])
            
        batch_loss = loss.item()
        total_test_loss += batch_loss       
        
        num_examples += b_input_ids.size(0) 

    avg_test_loss = total_test_loss / len(test_dataloader)

    test_acc = correct / num_examples
    
    test_time = format_time(time.time() - t0)    

    print("  Test Loss: {0:.6f}".format(avg_test_loss))
    print("  Test Accuracy: {0:.2f}".format(test_acc))
    print("  Test took: {:}".format(test_time))


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
            token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)
            reference_ids = token_reference.generate_reference(seq_len, device=device)
            eos_idx = input_ids.squeeze().detach().cpu().tolist().index(tokenizer.eos_token_id)
            reference_ids[eos_idx] = tokenizer.eos_token_id
            #print("eos_idx:", eos_idx)
            #print("input_ids:", input_ids.detach().tolist())
            #print("reference_ids:", reference_ids.detach().tolist())
            reference_layer = self.layer(reference_ids.unsqueeze(0))
            return reference_layer
        elif baseline_method == "gaussblur":
            reference_layer = self.layer(input_ids)
            blurrer = torchvision.transforms.GaussianBlur([19, 19], sigma=(9.5, 10.0))
            reference_layer[:,attr_mask,:] = blurrer(reference_layer[:,attr_mask,:])
            return reference_layer
        
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
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        inps, zeros = _format_input_baseline(inputs, None)
        _validate_input(inps, zeros, n_steps, method)

        zeros = _tensorize_baseline(inps, zeros)
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
                        return scattered_inputs_dict[device][
                            num_outputs_cumsum[layer_idx] : num_outputs_cumsum[
                                layer_idx + 1
                            ]
                        ]

                    return scattered_inputs_dict[device][num_outputs_cumsum[layer_idx]]

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
            # TODO: correct start point
            start_point, end_point = zeros, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_outputs(isinstance(self.layer, list), output), delta
        return _format_outputs(isinstance(self.layer, list), output)

def attribute(model, tokenizer):
    # ========================================
    #               Attribution
    # ========================================

    print("")
    print("Running Attribution...")
    model.to(device)
    
    t0 = time.time()

    model.eval()

    def model_attr_func(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        return model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask).logits[:,0]#.max(1).values
    
    
    ig = T5IntegratedGradients(model_attr_func, model.encoder.embed_tokens)
    #ig = IntegratedGradients(model_attr_func)
    
    '''
    def summarize_attributions(attributions, exclude_outliers=True):
        attributions = attributions.sum(dim=-1).squeeze(0)
        if exclude_outliers:
            p = torch.quantile(attributions, torch.tensor([0.05, 0.95], dtype = attributions.dtype, device= device))
            norm = torch.norm(attributions[(attributions > p[0])&(attributions < p[1])])
        else:
            norm = torch.norm(attributions)
        attributions = attributions / norm
        return attributions
    '''
    def summarize_attributions(attributions):
        attributions = attributions.sum(dim=-1).squeeze(0)
        #norm = torch.norm(attributions)
        #attributions = attributions / norm
        return attributions


    html_data = []

    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2]
        if model_type.startswith("t5") or model_type.startswith("google/t5"):
            b_label_masks = batch[3].to(device)
        
        batch_size = b_input_ids.size(0)
        with torch.no_grad():     
            
            if output_type == "one-hot":
                '''
                outputs = model(  b_input_ids,
                          attention_mask = b_masks,
                          token_type_ids=None,
                          labels = b_labels.to(device)
                        )
                '''
                raise NotImplementedError()
            elif output_type == "text":
                if model_type.startswith("t5") or model_type.startswith("google/t5"):
                    '''
                    outputs = model(
                                b_input_ids,
                                attention_mask=b_masks,
                                labels=b_labels.to(device)
                            )
                    '''
                    #decode_start_token_ids = torch.tensor([model.config.decoder_start_token_id]*b_input_ids.size(0)).unsqueeze(1).to(device)
                    target_encodings_dict = tokenizer(['<pad>']*batch_size, truncation=True, max_length=seq_len, padding="max_length")
                    decode_start_token_ids = torch.tensor(target_encodings_dict['input_ids']).to(device)
                    #print("decode_start_token_ids:", decode_start_token_ids)
                    decode_start_attention_mask = torch.tensor(target_encodings_dict['attention_mask']).to(device)
                    #print("decode_start_attention_mask:", decode_start_attention_mask)
                    
                    
                    
                    for i in range(batch_size):
                        attr_mask = ~((b_input_ids[i] == tokenizer.pad_token_id) | (b_input_ids[i] == tokenizer.eos_token_id))
                        #print("viz_mask:", viz_mask.detach().tolist())
                        
                        indices = b_input_ids[i].detach().tolist()
                        all_tokens = tokenizer.convert_ids_to_tokens(indices)
                        
                        scores = model(
                                input_ids=b_input_ids[i].unsqueeze(0),
                                attention_mask=b_masks[i].unsqueeze(0),
                                decoder_input_ids=decode_start_token_ids[i].unsqueeze(0),
                                decoder_attention_mask=decode_start_attention_mask[i].unsqueeze(0)
                            ).logits[:,0]
                        

                        attributions, delta = ig.attribute(inputs=b_input_ids[i].unsqueeze(0),
                                      baseline_method = baseline_method, #generate_reference(b_input_ids[i]).unsqueeze(0),
                                      attr_mask = attr_mask, 
                                      tokenizer = tokenizer, 
                                      additional_forward_args=(b_masks[i].unsqueeze(0), decode_start_token_ids[i].unsqueeze(0), decode_start_attention_mask[i].unsqueeze(0)),
                                      target = b_labels[i][0], method=attr_method,attribute_to_layer_input=False,
                                      return_convergence_delta=True,internal_batch_size=8,n_steps=200)
                        #print("attributions:", attributions)
                        #print("delta:", delta)
                        
                        attributions_sum = summarize_attributions(attributions)
                        
                        print("attributions_sum:", attributions_sum.detach().tolist())
                        #print("attributions_sum.sum():", attributions_sum.sum())

                        position_vis = viz.VisualizationDataRecord(
                            attributions_sum[attr_mask],
                            torch.max(torch.softmax(scores[0], dim=0)),
                            tokenizer.decode(torch.argmax(scores), skip_special_tokens=True),
                            tokenizer.decode(b_labels[i], skip_special_tokens=True),
                            tokenizer.decode(b_labels[i], skip_special_tokens=True),
                            attributions_sum.sum(),
                            np.array(all_tokens)[attr_mask.cpu()],
                            delta)
                        
                        viz_html = viz.visualize_text([position_vis]).data
                        print("viz_html:", viz_html)
                        html_data.append(viz_html)
                        
                        
                elif model_type.startswith("gpt2"):
                    '''
                    outputs = model(  b_input_ids,
                          attention_mask = b_masks,
                          token_type_ids=None,
                          labels = b_input_ids
                        )
                    '''
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    html_data = '\n'.join(html_data)
    
    with codecs.open('%s.html'%attr_method, 'w', 'utf-8') as file:
        file.write(html_data)
    
    attribute_time = format_time(time.time() - t0)    

    print("  Attribution took: {:}".format(attribute_time))



if __name__ == '__main__':
    
    doc_seqs, labels = load_data()
    tokenizer = load_tokenizer()
    if model_type.startswith("gpt2"):
        dataset = Dataset(doc_seqs, labels, tokenizer, max_length=seq_len)
    else:
        dataset = Dataset(doc_seqs, labels, tokenizer, max_length=seq_len)
    train_dataloader, validation_dataloader, test_dataloader = init_data_loader(dataset)
    '''
    model = load_model(tokenizer)
    print_model_params()
    
    training_stats = train(model, epochs, learning_rate, "fine-tuning", tokenizer)
    plot_training_stats(training_stats)

    model = load_model_from_local_path('./model_save/', tokenizer)
    test(model, tokenizer)
    '''
    model = load_model_from_local_path('./model_save/', tokenizer)
    attribute(model, tokenizer)
    
