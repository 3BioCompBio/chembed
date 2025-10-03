# Metrics from MOSES: (Molecular sets (MOSES): a benchmarking platform for molecular generation models, Polykovskiy, 2020)
# Functions are from their github: https://github.com/molecularsets/moses

from typing import Sequence, Union
from collections.abc import Iterable
from numpy.typing import NDArray

import torch
import numpy as np

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """ 
    From MOSES, I didn't touch it
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1 
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)


def internal_diversity_from_fingerprints(gen_fps: Union[Sequence, NDArray], p: int, device: Union[torch.device, str]) -> float:
    if not isinstance(gen_fps, np.ndarray):
        if not isinstance(gen_fps, list):
            if not isinstance(gen_fps, Iterable):
                raise ValueError("gen_fps is not iterable")
            else:
                gen_fps = list(gen_fps) 
        gen_fps = np.array(gen_fps)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps, agg='mean', p=p, device=device)).mean().item()


def IntDiv1(gen_fps: Union[Sequence, NDArray], device) -> float:
    return internal_diversity_from_fingerprints(gen_fps, device=device, p=1)

def IntDiv2(gen_fps: Union[Sequence, NDArray], device) -> float:
    return internal_diversity_from_fingerprints(gen_fps, device=device, p=2)
