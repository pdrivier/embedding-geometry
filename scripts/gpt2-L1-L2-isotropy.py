import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns

from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

## Define custom IsoScore functions from Rudman et al. 2022
## Step 2
def pca_normalization(points):
    """points: (m samples x n dimensions)"""
    
    pca = PCA(n_components=len(np.transpose(points)))
    points = pca.fit_transform(points)

    return np.transpose(points)

## Step 3
def get_diag_of_cov(points):
    """points: (n dims x m samples)"""
    
    n = np.shape(points)[0]
    cov = np.cov(points)
    cov_diag = cov[np.diag_indices(n)]

    return cov_diag

## Step 4
def normalize_diagonal(cov_diag):

    n = len(cov_diag)
    cov_diag_normalized = (cov_diag*np.sqrt(n))/np.linalg.norm(cov_diag)

    return cov_diag_normalized

## Step 5
def get_isotropy_defect(cov_diag_normalized):

    n = len(cov_diag_normalized)
    iso_diag = np.eye(n)[np.diag_indices(n)]
    l2_norm = np.linalg.norm(cov_diag_normalized - iso_diag)
    normalization_constant = np.sqrt(2*(n-np.sqrt(n)))
    isotropy_defect = l2_norm/normalization_constant

    return isotropy_defect

## Interlude
def get_kdims(isotropy_defect, embed_dim): 
    
    n = embed_dim
    k = ((n-(isotropy_defect**2)*(n-np.sqrt(n)))**2) / n
    
    return k

def get_fraction_dims(k, embed_dim):
    
    n = embed_dim
    phi = k/n
    
    return phi

## Step 6
def get_IsoScore(isotropy_defect, embed_dim):

    n = embed_dim
    the_score = ((n-(isotropy_defect**2)*(n-np.sqrt(n)))**2-n)/(n*(n-1))

    return the_score


### Define dataset class

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)



## Cat's Biliingual models:
# Checkpoints are taken at training steps: 0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000.
CHECKPOINTS = [0, 10000, 20000, 30000, 40000, 50000, 64000, 64010, 64020, 64030, 64040, 64050, 64060, 64070, 64080, 64090, 64100, 64110, 64120, 64130, 64140, 64150, 64160, 64170, 64180, 64190, 64200, 64300, 64400, 64500, 64600, 64700, 64800, 64900, 65000, 66000, 67000, 68000, 69000, 70000, 80000, 90000, 100000, 110000, 120000, 128000]

# Define a machine device to allocate model to
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


### TEST CODE: 

# mpath = "catherinearnett/B-GPT_en_nl_simultaneous"
# tokenizer = AutoTokenizer.from_pretrained(mpath)
# checkpoint = "128000"
# model = AutoModel.from_pretrained(mpath, revision = checkpoint).to(device)

# # Grab input embeddings, shape (vocab_size x embed_dim)
# input_embed = model.wte.weight

# # Compute the IsoScore for this matrix
# pca_embed = pca_normalization(input_embed.cpu().detach())
# diag_embed_cov = get_diag_of_cov(pca_embed)
# normdiag_embed_cov = normalize_diagonal(diag_embed_cov)
# isotropy_defect = get_isotropy_defect(normdiag_embed_cov)

# kdims = get_kdims(isotropy_defect, input_embed)
# phi = get_fraction_dims(kdims, input_embed)
# isoscore = get_IsoScore(isotropy_defect, input_embed)


# TODO: 
# Iterate through checkpoints for single language model, grab input embedding matrix, and 
# compute isoscore

mpath = "catherinearnett/B-GPT_en_nl_simultaneous"
gather = []
for checkpoint in tqdm(CHECKPOINTS):

    tokenizer = AutoTokenizer.from_pretrained(mpath)
    model = AutoModel.from_pretrained(mpath, revision = str(checkpoint)).to(device)
    mname = mpath.split("/")[1] + "-check-" 

    input_embed = model.wte.weight
    embed_dim = input_embed.shape[1]

    # Compute the IsoScore for this matrix
    pca_embed = pca_normalization(input_embed.cpu().detach())
    diag_embed_cov = get_diag_of_cov(pca_embed)
    normdiag_embed_cov = normalize_diagonal(diag_embed_cov)
    isotropy_defect = get_isotropy_defect(normdiag_embed_cov)

    kdims = get_kdims(isotropy_defect, embed_dim)
    phi = get_fraction_dims(kdims, embed_dim)
    isoscore = get_IsoScore(isotropy_defect, embed_dim)

    # Populate a dictionary with the isotropy measures
    gather.append({"model": mname, 
            "kdims": kdims,
            "isoscore": isoscore,
            "checkpoint": checkpoint
        })

df = pd.DataFrame(gather)

sns.set(style="whitegrid",font_scale=1.2)
sns.lineplot(data=df,x="checkpoint",y="kdims")
plt.show()








