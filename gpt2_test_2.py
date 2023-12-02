'''
Created on Oct. 20, 2022

@author: Yihao Fang
'''

'''
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5))
'''

'''
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
last_hidden_state = output.last_hidden_state
print(last_hidden_state.shape)
'''


'''
import torch
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2DoubleHeadsModel.from_pretrained("gpt2")

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
# Update the model embeddings with the new vocabulary size
embedding_layer = model.resize_token_embeddings(len(tokenizer))

choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]

print(encoded_choices)

cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

print(cls_token_location)

input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

print(input_ids.shape)
print(mc_token_ids.shape)

outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_logits = outputs.logits
mc_logits = outputs.mc_logits

print(lm_logits.shape)
print(mc_logits.shape)
'''

'''
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add a [CLS] to the vocabulary (we should train it also!)
#num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
# Update the model embeddings with the new vocabulary size
#embedding_layer = model.resize_token_embeddings(len(tokenizer))

inputs = tokenizer("Hello, my dog is cute. His age is 45.", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits

print(loss)
print(logits.shape)

generated_ids = model.generate(inputs["input_ids"], do_sample=True, num_return_sequences=5, max_length=200)

print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
'''
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

from transformers import GPT2Tokenizer, GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel, GPT2ForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')

import re

def is_nan(x):
    return (x != x)


def load_data():
    # load into a data frame
    filename = 'data/iTrade_dataset.csv'
    iTrade_dataset_df = pd.read_csv(filename, encoding='utf-8', parse_dates=['Week'])  
    
    iTrade_columns = list(iTrade_dataset_df)
    
    #print(iTrade_columns)
    
    num_iTrade_htmls = int(re.search(r'Text (\d+?)\-\d', iTrade_columns[-2]).group(1))
    
    #numerical_column_names = ['Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV', 'Awareness', 'Consideration', 'Purchase']
    numerical_column_names = ['Digital', 'DigitalOOH', 'OOH', 'Print', 'Radio', 'SEM', 'SEM 2', 'Social', 'TV']
    
    docs = []
    for idx, row in iTrade_dataset_df.iterrows():
        
        numerals = []
        # include marketing spending
        for numerical_column_name in numerical_column_names:
            numerals.append(str(row[numerical_column_name]) if not is_nan(row[numerical_column_name]) else "")
        
        texts = []
        for i in range(num_iTrade_htmls):
            score_column_name = 'HTML %d Page Views'%(i+1)
            page_texts = []
            for j in range(4):
                text_column_name = 'Text %d-%d'%(i+1, j+1)
                if text_column_name in iTrade_columns:
                    page_texts.append(row[text_column_name] if not is_nan(row[text_column_name]) else "")
            page_score = row[score_column_name] if score_column_name in iTrade_columns and not is_nan(row[score_column_name]) else 0
            texts.append(('\n'.join(page_texts), page_score))
        k = 5
        top_k_idxes = sorted(sorted(range(len(texts)), key = lambda idx: texts[idx][1])[-k:])
        
        # include top-5 page views
        # numerals.extend([str(texts[idx][1]) for idx in top_k_idxes])
        
        #print("top_k_idxes:", top_k_idxes)
        docs.append(' '.join(numerals) + '\n' + ('\n'.join([texts[idx][0] for idx in top_k_idxes])))
        #docs.append(' '.join(numerals))
        #docs.append('\n'.join([texts[idx][0] for idx in top_k_idxes]))

    doc_lengths = []
    
    for doc in docs:
    
        # get rough token count distribution
        tokens = nltk.word_tokenize(doc)
    
        doc_lengths.append(len(tokens))
    
    doc_lengths = np.array(doc_lengths)
    
    sns.displot(doc_lengths)
    plt.show()
    

    labels = []
    
    pre_conversion_rate = 0
    for idx, row in iTrade_dataset_df.iterrows():
        
        labels.append(1 if row['Purchase']/row['Consideration'] > pre_conversion_rate else 0)
        pre_conversion_rate = row['Purchase']/row['Consideration']

    doc_queue = []
    
    doc_seqs = []
    
    num_weeks = 2
    
    for doc in docs:
        doc_queue.append(doc)
        if len(doc_queue)==num_weeks:
            doc_seqs.append('\n'.join(doc_queue))
            doc_queue.pop(0)
    
    labels = labels[num_weeks-1:]
    
    return doc_seqs, labels
            
doc_seqs, labels = load_data()

def load_tokenizer():
    # Load the GPT tokenizer.
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add a [CLS] to the vocabulary (we should train it also!)
    #num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    
    print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
    print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
    print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
    print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

    return tokenizer

tokenizer = load_tokenizer()



class GPT2Dataset(Dataset):

    def __init__(self, texts, labels, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = labels
        #self.cls_token_locations = []
    
        for text in texts:
    
            encodings_dict = tokenizer('<|startoftext|>'+ text + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            #print(tokenizer.decode(encodings_dict['input_ids']))
            #self.cls_token_locations.append(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            #print(torch.tensor(encodings_dict['input_ids'].index(tokenizer.cls_token_id)))
            
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        #return self.input_ids[idx], self.attn_masks[idx], self.cls_token_locations[idx], self.labels[idx]
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


dataset = GPT2Dataset(doc_seqs, labels, tokenizer, max_length=768)


def init_data_loader():
    # Split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    
    batch_size = 8
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
    
    return train_dataloader, validation_dataloader

train_dataloader, validation_dataloader = init_data_loader()



def load_model():
    # I'm not really doing anything with the config buheret
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    #config.summary_use_proj = True
    #config.summary_proj_to_labels = True
    config.pad_token_id = tokenizer.pad_token_id
    # instantiate the model
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=config)
    
    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    return model

model = load_model()

device = torch.device("cuda")

def train(model):
    # Tell pytorch to run this model on the GPU.
    
    model.cuda()
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    
    # some parameters I cooked up that work reasonably well
    
    epochs = 6
    learning_rate = 1e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    
    # this produces sample output every 100 steps
    sample_every = 100
    
    
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
    
    
    def format_time(elapsed):
        return str(datetime.timedelta(seconds=int(round((elapsed)))))
    

    total_t0 = time.time()
    
    training_stats = []
    
    model = model.to(device)
    
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
            b_mc_token_ids = batch[2].to(device)
            b_labels = batch[2].to(device)
    
            model.zero_grad()        
            
            outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels
                            )
    
            loss = outputs.loss
            '''
            outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              mc_token_ids=b_mc_token_ids, 
                              mc_labels = b_labels
                            )
            
            
            lm_logits = outputs.logits
            mc_logits = outputs.mc_logits
            
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(mc_logits, b_labels.float())
            print(b_labels)
            print(mc_logits)
            
            loss = outputs.mc_loss
            '''
            
            #loss = outputs[0]  
    
            batch_loss = loss.item()
            total_train_loss += batch_loss
    
            # Get sample every x batches.
            if step % sample_every == 0 and not step == 0:
    
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
    
                model.eval()
    
                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 200,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()
    
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
        nb_eval_steps = 0
        correct = 0
        
        num_examples = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_mc_token_ids = batch[2].to(device)
            b_labels = batch[2].to(device)
            
            
            with torch.no_grad():        
                '''
                outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              mc_token_ids=b_mc_token_ids
                            )
            
                
                lm_logits = outputs.logits
                mc_logits = outputs.mc_logits
                
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(mc_logits, b_labels.float())
                
                print(mc_logits)
                '''
                outputs = model(  b_input_ids,
                              attention_mask = b_masks,
                              token_type_ids=None,
                              labels = b_labels
                            )
    
                loss = outputs.loss
                
                
                preds = outputs.logits.argmax(dim=1)
                
                #print("b_input_ids:", [tokenizer.decode(input_ids) for input_ids in b_input_ids])
                #print("outputs.logits:", outputs.logits)
                #print("preds:", preds)
                #print("b_labels:", b_labels)              
                #print(preds.eq(b_labels), preds.eq(b_labels).sum().item())
                    
                

            correct += preds.eq(b_labels).sum().item()
                
            batch_loss = loss.item()
            total_eval_loss += batch_loss       
            
            num_examples += b_input_ids.size(0) 
    
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        correct = correct / num_examples
        
        validation_time = format_time(time.time() - t0)    
    
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation Accuracy: {0:.2f}".format(correct))
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

#save_model()

def generate_output():

    model.eval()
    
    prompt = "My dog is a social animal."
    
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)
    
    print(generated)
    
    sample_outputs = model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 300,
                                    top_p=0.95, 
                                    num_return_sequences=3
                                    )
    
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
