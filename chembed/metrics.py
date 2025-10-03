from typing import Sequence, List, Union
from numpy.typing import NDArray

import numpy as np
from sklearn import manifold

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from chembed.errors import warn

def reconstruction_loss(logits: Tensor, tokens: Tensor) -> Tensor:
    return F.cross_entropy(logits.permute(0,2,1), tokens, reduction='none').mean()

def reconstruction_loss_unreduced(logits: Tensor, tokens: Tensor) -> Tensor:
    return F.cross_entropy(logits.permute(0,2,1), tokens, reduction='none').sum().item()
    
def kl_divergence_loss(mu: Tensor, sigma: Tensor) -> Tensor:
    sigma2 = sigma.pow(2)
    return 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).mean()
    
def kl_divergence_loss_unreduced(mu: Tensor, sigma: Tensor) -> Tensor:
    sigma2 = sigma.pow(2)
    return 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).sum().item()

def properties_loss(predicted_properties: Tensor, target_properties: Tensor) -> Tensor:
    return nn.MSELoss(reduction='mean')(predicted_properties, target_properties)

def token_reconstruction_accuracy(output_tokens: Tensor, target_tokens: Tensor) -> Tensor:
    assert(output_tokens.shape==target_tokens.shape)
    return (output_tokens == target_tokens).float().mean()

def string_reconstruction_accuracy(output_tokens: Tensor, target_tokens: Tensor) -> Tensor:
    return (output_tokens == target_tokens).all(dim=1).float().mean(dim=0)

def string_reconstruction_accuracy_sum(output_tokens: Tensor, target_tokens: Tensor) -> Tensor:
    return (output_tokens == target_tokens).all(dim=1).float().sum(dim=0)

def MSE_unreduced(predicted_properties: Tensor, target_properties: Tensor) -> Tensor:
    return nn.MSELoss(reduction='none')(predicted_properties, target_properties).sum(dim=0)

def compute_euclidean_distance_matrix(zs: Tensor) -> Tensor:
    if zs.ndim != 2:
        zs = zs.flatten(start_dim=1)
    return torch.cdist(zs, zs)

def compute_tanimoto_similarity_matrix(fps: Tensor) -> Tensor:
    intersections = ((fps.unsqueeze(1) * fps.unsqueeze(0)) > 0).sum(dim=-1)
    unions = ((fps.unsqueeze(1) + fps.unsqueeze(0)) > 0).sum(dim=-1)
    return intersections/unions

def compute_cos_similarity_matrix(zs: Tensor) -> Tensor:
    zs_normalized = F.normalize(zs)
    return zs_normalized @ zs_normalized.T

def compute_rescaled_cos_similarity_matrix(zs: Tensor) -> Tensor:
    return 0.5*(1+compute_cos_similarity_matrix(zs))

def tanimoto_cos_loss(zs: Tensor, fps: Tensor) -> Tensor:
    warn(DeprecationWarning, "This loss function, used in early versions, will be removed.") 
    M_tanimoto = compute_tanimoto_similarity_matrix(fps)
    M_cos = compute_rescaled_cos_similarity_matrix(zs.flatten(start_dim=1))
    return (M_tanimoto-M_cos).pow(2).sum() / M_tanimoto.nelement() 

def tanimoto_euclidean_correlation_loss(zs: Tensor, fps: Tensor) -> Tensor:
    warn(DeprecationWarning, "This loss function, used in early versions, will be removed.") 
    M_tanimoto = compute_tanimoto_similarity_matrix(fps)
    D_euclidean = compute_euclidean_distance_matrix(zs.flatten(start_dim=1))
    corr_matrix = torch.corrcoef(torch.stack([M_tanimoto.flatten(), D_euclidean.flatten()]))
    return corr_matrix[0,1]

def tanimoto_shifted_euclidean_correlation_loss(zs: Tensor, fps: Tensor) -> Tensor:
    warn(DeprecationWarning, "This loss function, used in early versions, will be removed.") 
    unshifted_loss = tanimoto_euclidean_correlation_loss(zs, fps)
    return 1+unshifted_loss

def tanimoto_euclidean_correlation_triu_loss(zs: Tensor, fps: Tensor) -> Tensor:
    M_tanimoto = compute_tanimoto_similarity_matrix(fps)
    D_euclidean = compute_euclidean_distance_matrix(zs.flatten(start_dim=1))
    mask = torch.triu(torch.ones_like(D_euclidean, dtype=bool), diagonal=1)
    ts = M_tanimoto[mask].flatten()
    ds = D_euclidean[mask].flatten()
    corr_matrix = torch.corrcoef(torch.stack([ts, ds]))
    return corr_matrix[0,1]

def tanimoto_shifted_euclidean_correlation_triu_loss(zs: Tensor, fps: Tensor) -> Tensor:
    unshifted_loss = tanimoto_euclidean_correlation_triu_loss(zs, fps)
    return 1+unshifted_loss

def uniqueness(smiles_list: Sequence[str]) -> float:
    if not smiles_list:
        return 0.0
    return len(set(smiles_list)) / float(len(smiles_list))

def trustworthiness_score(zs: Union[Tensor, NDArray], fps: Union[Tensor, NDArray, List], n_neighbors: int) -> float:
    if isinstance(zs, Tensor):
        zs = zs.detach().cpu()
        if zs.ndim == 3:
            zs = zs.squeeze(1)
        elif zs.ndim != 2:
            raise ValueError(f"zs.ndim={zs.ndim}")
    if isinstance(fps, Tensor):
        fps = fps.detach().cpu()
        if fps.ndim != 2:
            raise ValueError(f"fps.ndim={fps.ndim}")
    if not (isinstance(fps, Tensor) or isinstance(fps, np.ndarray)):
        fps = np.stack([np.array(fp) for fp in fps])
    return manifold.trustworthiness(fps, zs, n_neighbors=n_neighbors, metric='jaccard')

def compute_L1_distance_matrix(props: List[float]) -> NDArray:
    N = len(props)
    D = np.zeros((N,N))
    for i in range(N-1):
        for j in range(i+1,N):
            D[i,j] = abs(props[i]-props[j])
            D[j,i] = D[i,j]
    return D
