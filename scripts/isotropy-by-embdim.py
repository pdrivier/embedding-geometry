import os
import shutil
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



## Do this in steps if you're on your local machine with small storage
## NOTE: deduplication doesn't seem to make a difference for the input embedding dimensionality

# MODELS = ["EleutherAI/pythia-14m",
#           "EleutherAI/pythia-70m",
#           "EleutherAI/pythia-160m",
#           "EleutherAI/pythia-410m",
#           "EleutherAI/pythia-1b",
#           "EleutherAI/pythia-1.4b"
#           ]

MODELS = ["EleutherAI/pythia-2.8b",
          "EleutherAI/pythia-6.9b"]


# Define a machine device to allocate model to
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# cache_path = "~/.cache/huggingface/hub/"

gather = []
for mpath in tqdm(MODELS):

    ## Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mpath)

    ## Grab some model details to store later
    mname = mpath.split("/")[1]
    msize = mname.split("-")[1]
    dedup = mname.split("-")[-1]

    ## Check: is this for a model trained on deduplicated dataset?
    if dedup in msize:
        dedup = 0
    else:
        dedup = 1
        
    ## Shunt the model to the gpu if available
    model = AutoModel.from_pretrained(mpath).to(device)

    ## Get input embedding matrix
    input_embed = model.get_input_embeddings().weight
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
        "deduplicated": dedup,
        "model_size": msize,
        "embed_dim": embed_dim,
        "kdims": kdims,
        "fraction_dims": phi,
        "isoscore": isoscore})

    ## TODO: Clear the model from the cache to avoid local space issues
    # cached_folders = os.listdir(cache_path)
    # model_folder = [f for f in cached_folders if f.startswith("models--")]
    # shutil.rmtree(model_folder)



df = pd.DataFrame(gather)

savepath = "pythia-inputemb-results/"
if not os.path.exists(savepath):
    os.mkdir(savepath)

filename = "pythia-" + MODELS[0].split("pythia-")[1] + "-to-" + MODELS[-1].split("pythia-")[1] + ".csv"

df.to_csv(os.path.join(savepath,filename))
