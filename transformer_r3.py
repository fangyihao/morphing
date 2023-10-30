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
torch.manual_seed(42)

from transformers import GPT2Tokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2PreTrainedModel, GPT2ForSequenceClassification
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

import nltk
nltk.download('punkt')

import re
from statistics import mode
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

#hyper-parameters
model_type = "gpt2"  #"bert-base-uncased"
k = 5  # top-k pages
batch_size = 8
num_labels = 3   # number of classes to predict 0: conversion value is not volatile 1: conversion value is volatile and increasing 2: conversion value is volatile and decreasing
epochs = 30
learning_rate = 1e-4
warmup_steps = 1e2
epsilon = 1e-8
seed_val = 42

def is_nan(x):
    return (x != x)


def load_data():

    filename = 'data/iTrade_dataset.csv'
    iTrade_dataset_df = pd.read_csv(filename, encoding='utf-8', parse_dates=['Week'])  
    
    iTrade_columns = list(iTrade_dataset_df)
    
    num_iTrade_htmls = int(re.search(r'Text (\d+?)\-\d', iTrade_columns[-2]).group(1))
    spending_column_names = ['Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV']
    
    docs = []
    for idx, row in iTrade_dataset_df.iterrows():
        
        spendings = []
        # include marketing spending
        for spending_column_name in spending_column_names:
            spendings.append(str(row[spending_column_name]) if not is_nan(row[spending_column_name]) else "")
        
        pages = []

        for i in range(num_iTrade_htmls):
            score_column_name = 'HTML %d Page Views'%(i+1)
            texts = []
            images = []
            for j in range(4):
                text_column_name = 'Text %d-%d'%(i+1, j+1)
                if text_column_name in iTrade_columns:
                    texts.append(row[text_column_name] if not is_nan(row[text_column_name]) else "")
                    
                image_column_name = 'Image Features %d-%d'%(i+1, j+1)
                if image_column_name in iTrade_columns:
                    images.append(row[image_column_name] if not is_nan(row[image_column_name]) else "")
                
            page_score = row[score_column_name] if score_column_name in iTrade_columns and not is_nan(row[score_column_name]) else 0
            pages.append(('\n'.join(texts), '\n'.join(images), page_score))
        
        top_k_idxes = sorted(sorted(range(len(pages)), key = lambda idx: pages[idx][2])[-k:])
        
        # include top-5 page views
        page_views = []
        page_views.extend([str(pages[idx][2]) for idx in top_k_idxes])
        
       
        #docs.append('\n'.join([spending_column_name + ":" + spending for spending_column_name, spending in zip(spending_column_names, spendings)]) + '\n' + ('\n'.join([pages[idx][0] for idx in top_k_idxes])))
        docs.append(' '.join(spendings) + '\n' + ('\n'.join([pages[idx][0] for idx in top_k_idxes])))
        #docs.append(' '.join(spendings) + '\n' + ('\n'.join([pages[idx][1] for idx in top_k_idxes])) + '\n' + ('\n'.join([pages[idx][0] for idx in top_k_idxes])))
        #docs.append('\n'.join([spending_column_name + ":" + spending for spending_column_name, spending in zip(spending_column_names, spendings)]) + '\n' + ' '.join(page_views))
        #docs.append('\n'.join([pages[idx][0] for idx in top_k_idxes]))
        #docs.append(' '.join(spendings))
        #docs.append(' '.join(page_views))

    labels = []
    
    pre_conversion = 0
    conversion_volatility = []
    for idx, row in iTrade_dataset_df.iterrows():
        conversion_volatility.append(abs(row['Purchase']- pre_conversion))
        pre_conversion = row['Purchase']
    conversion_volatile_threshold = np.median(conversion_volatility)
    print("conversion volatility mean, median, mode:", np.mean(conversion_volatility), np.median(conversion_volatility), mode(conversion_volatility))    
    for idx, row in iTrade_dataset_df.iterrows():
        labels.append(0 if abs(row['Purchase']- pre_conversion) < conversion_volatile_threshold else (1 if row['Purchase']>pre_conversion else 2))
        pre_conversion = row['Purchase']
    sns.displot(conversion_volatility)
    plt.show()

    doc_queue = []
    
    doc_seqs = []
    
    num_weeks = 2
    
    for doc in docs:
        doc_queue.append(doc)
        if len(doc_queue)==num_weeks:
            doc_seqs.append('\n'.join(doc_queue))
            doc_queue.pop(0)
            
    doc_lengths = []
    for doc in doc_seqs:
        tokens = nltk.word_tokenize(doc)
        doc_lengths.append(len(tokens))
    doc_lengths = np.array(doc_lengths)
    sns.displot(doc_lengths)
    plt.show()
    
    labels = labels[num_weeks-1:]
    
    return doc_seqs, labels
            
doc_seqs, labels = load_data()

def load_tokenizer():
    # Load tokenizer.
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    if model_type.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = BertTokenizer.from_pretrained(model_type)

    # Add a [CLS] to the vocabulary (we should train it also!)
    #num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    
    #print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
    #print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
    #print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
    #print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

    return tokenizer

tokenizer = load_tokenizer()


class Dataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = labels
        #self.cls_token_locations = []
    
        for text in texts:
            if model_type.startswith("gpt2"):
                encodings_dict = tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            else:
                encodings_dict = tokenizer(text, truncation=True, padding="max_length")
            
            #print(tokenizer.decode(encodings_dict['input_ids']))
            #self.cls_token_locations.append(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            #print(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]

if model_type.startswith("gpt2"):
    dataset = Dataset(doc_seqs, labels, tokenizer, max_length=1024)
else:
    dataset = Dataset(doc_seqs, labels, tokenizer)


def init_data_loader():

    # Split into training and validation sets
    train_size = round(0.7 * len(dataset))
    val_size = round(0.15 * len(dataset))
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

train_dataloader, validation_dataloader, test_dataloader = init_data_loader()



def load_model():
    if model_type.startswith("gpt2"):
        config = GPT2Config.from_pretrained(model_type, output_hidden_states=False)
    else:
        config = BertConfig.from_pretrained(model_type, output_hidden_states=False)
    config.num_labels = num_labels
    #config.summary_use_proj = True
    #config.summary_proj_to_labels = True
    config.pad_token_id = tokenizer.pad_token_id
    # instantiate the model
    if model_type.startswith("gpt2"):
        model = GPT2ForSequenceClassification.from_pretrained(model_type, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(model_type, config=config)
    
    model.resize_token_embeddings(len(tokenizer))

    return model

def load_model_from_local_path(model_path):
    if model_type.startswith("gpt2"):
        config = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
    else:
        config = BertConfig.from_pretrained(model_path, output_hidden_states=False)
    print("config.num_labels:", config.num_labels)
    #config.summary_use_proj = True
    #config.summary_proj_to_labels = True
    config.pad_token_id = tokenizer.pad_token_id
    # instantiate the model
    if model_type.startswith("gpt2"):
        model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(model_path, config=config)
    
    model.resize_token_embeddings(len(tokenizer))

    return model
    

model = load_model()

device = torch.device("cuda")


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
    

def train(model):
    # Tell pytorch to run this model on the GPU.
    model.cuda()
    
    if model_type.startswith("gpt2"):
        for name, param in model.transformer.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
        
    
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
            b_labels = batch[2].to(device)
    
            model.zero_grad()        
            
            outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels
                            )
    
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
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
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
            b_labels = batch[2].to(device)
            
            with torch.no_grad():        
                
                outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels
                            )
    
                loss = outputs.loss
                
                
                preds = outputs.logits.argmax(dim=1)

            correct += preds.eq(b_labels).sum().item()
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss       
            
            num_examples += b_input_ids.size(0) 
    
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        val_acc = correct / num_examples
        
        validation_time = format_time(time.time() - t0)    
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
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

training_stats = train(model)


def plot_training_stats():
    
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    
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
    
plot_training_stats()

def print_model_params():
    
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    
    print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))
    
    print('==== Embedding Layer ====\n')
    
    for p in params[0:2]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== First Transformer ====\n')
    
    for p in params[2:14]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
    print('\n==== Output Layer ====\n')
    
    for p in params[-2:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print_model_params()
    

model = load_model_from_local_path('./model_save/')

def test(model):
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")
    model.cuda()
    
    t0 = time.time()

    model.eval()

    total_test_loss = 0

    correct = 0
    
    num_examples = 0
    # Evaluate data for one epoch
    for batch in test_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_masks = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        
        with torch.no_grad():        
            
            outputs = model(  b_input_ids,
                          attention_mask = b_masks,
                          token_type_ids=None,
                          labels = b_labels
                        )

            loss = outputs.loss
            
            
            preds = outputs.logits.argmax(dim=1)
            

        correct += preds.eq(b_labels).sum().item()
            
        batch_loss = loss.item()
        total_test_loss += batch_loss       
        
        num_examples += b_input_ids.size(0) 

    avg_test_loss = total_test_loss / len(test_dataloader)

    test_acc = correct / num_examples
    
    test_time = format_time(time.time() - t0)    

    print("  Test Loss: {0:.2f}".format(avg_test_loss))
    print("  Test Accuracy: {0:.2f}".format(test_acc))
    print("  Test took: {:}".format(test_time))

test(model)

