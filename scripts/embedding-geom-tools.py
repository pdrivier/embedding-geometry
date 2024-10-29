import os
import shutil

from sklearn.decomposition import PCA



### Define custom function for clearing your model cache, if computer is 
# 	storage-constrained
def clear_cache(): 

	return 



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


def isoscore_wrap(points, embed_dim): 

	pca_points = pca_normalization(points)
	cov_diag = get_diag_of_cov(pca_points)
	cov_diag_normalized = normalize_diagonal(cov_diag)
	isotropy_defect = get_isotropy_defect(cov_diag_normalized)
	k = get_kdims(isotropy_defect, embed_dim)
	phi = get_fraction_dims(k, embed_dim)
	the_score = get_IsoScore(isotropy_defect, embed_dim)

	return k, phi, the_score



