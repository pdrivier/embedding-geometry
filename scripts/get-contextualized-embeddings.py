
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



### TODO: 
######## Select an established corpus and language (e.g. OSCAR - english; Europarl - english; CLEAR Corpus)
######## Get sentences
######## Get contextualized embeddings from the sentences for each layer of the model

# access_token = ## this is unique to each user, find wherever you keep your tokens or make a new one 
# os.environ["HF_TOKEN"] = access_token


datapath = "data/"
corpora = os.listdir(datapath)
corpus = corpora[0]


df_corpus = pd.read_csv(os.path.join(datapath,corpus))
excerpts = df_corpus["Excerpt"].values
def split_excerpts(list_of_excerpts): 

	list_of_lines = []
	for excerpt in list_of_excerpts:

		lines = sent_tokenize(excerpt)

		# Store your lines in your list
		list_of_lines.append(lines)

	# Flatten your list of lines
	flat_list = [item for sublist in list_of_lines for item in sublist]

	return flat_list


### Get sentences from corpus
sentences = split_excerpts(excerpts)

### TODO: Getting random subset
random_subset = random.sample(sentences, 100)

### TODO: Get contextualized embeddings for each sentence and then subset (randomly?) a word from each sentence


