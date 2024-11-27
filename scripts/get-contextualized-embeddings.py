
import functools
import os
import shutil
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import random

from datasets import load_dataset, load_from_disk
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

### Custom functions to ...
### ... find the target tokens within tokenized sequence
def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None

@functools.lru_cache(maxsize=None)  # This will cache results, handy later...


### ... grab the embeddings for your target tokens
def get_embedding(model, tokenizer, sentence, target, layer, device):
    """Get a token embedding for target in sentence"""
    
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Run model
    with torch.no_grad():
        output = model(**inputs,output_hidden_states=True)
        hidden_states = output.hidden_states

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding


### TODO: 
######## Select an established corpus and language (e.g. OSCAR - english; Europarl - english; CLEAR Corpus)
######## Get sentences
######## Get contextualized embeddings for a target word from the sentences for each layer of the model

# access_token = ## this is unique to each user, find wherever you keep your tokens or make a new one 
# os.environ["HF_TOKEN"] = access_token


datapath = "data/"
corpora = os.listdir(datapath)
corpus = corpora[0]


df_corpus = pd.read_csv(os.path.join(datapath,corpus))
excerpts = df_corpus["Excerpt"].values
def split_excerpts(list_of_excerpts): 

	list_of_lines = []
	for excerpt in tqdm(list_of_excerpts):

		lines = sent_tokenize(excerpt)

		# Store your lines in your list
		list_of_lines.append(lines)

	# Flatten your list of lines
	flat_list = [item for sublist in list_of_lines for item in sublist]

	return flat_list


### Get sentences from corpus
sentences = split_excerpts(excerpts)

### TODO: Getting random subset
random_subset = random.sample(sentences, 1000)

#####
#####

### TODO: Get contextualized embeddings for each sentence and then subset (randomly?) a word from each sentence
MODELS = ["catherinearnett/B-GPT_en_pl_simultaneous"]
# Checkpoints are taken at training steps: 0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000.
CHECKPOINTS = [0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000]
# Define a machine device to allocate model to
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


mpath = MODELS[0]
checkpoint = CHECKPOINTS[5]
tokenizer = AutoTokenizer.from_pretrained(mpath)
model = AutoModel.from_pretrained(mpath, revision = str(checkpoint)).to(device)

# for each sentence, 
##### tokenize the sentence
##### randomly select a token index from the sequence,
##### grab the corresponding "target" token in text form, 
##### save the list of target words, with their sentences in a df


# Tokenize sentence
sentence = random_subset[30]
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False) #don't want to accidentally select special tokens, so tokenize first without them!
target_token_id = random.sample(inputs["input_ids"].tolist()[0],1)
target_token = tokenizer.decode(target_token_id)
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).to(device) #retokenize with special tokens
token_sequence = inputs["input_ids"][0].tolist()
target_sentence_id = [i for i,val in enumerate(token_sequence) if val == target_token_id[0]][0] #use this to find the position of the token index in the sentence


# Run model
with torch.no_grad():
    output = model(**inputs,output_hidden_states=True)
    hidden_states = output.hidden_states

# Iterate through layers, and grab the embedding for just the target token per layer
layer = 4
# Get layer
selected_layer = hidden_states[layer][0]

#grab just the embeddings for your target token
token_embedding = selected_layer[target_sentence_id:target_sentence_id+1]

