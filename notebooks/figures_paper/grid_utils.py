from typing import Tuple

import numpy as np

import torch
from torch import Tensor

def normalize(v: Tensor) -> Tensor:
    return v/torch.norm(v)

def get_random_orthonormal_basis_from_u(u: Tensor) -> Tuple[Tensor, Tensor]:
    w = normalize(torch.randn(size=u.shape))
    v_unnormalized = w - torch.dot(w.flatten(), u.flatten()) * u
    v = normalize(v_unnormalized)
    return u, v

def get_random_orthonormal_basis(d: int = 256) -> Tuple[Tensor, Tensor]:
    u = normalize(torch.randn(size=(1,d)))
    return get_random_orthonormal_basis_from_u(u)

def set_seed(random_seed: int) -> None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

def get_zs_on_grid(origin: Tensor, u: Tensor, v: Tensor, N_x: int, N_y: int, step_size: int) -> Tensor:
    x = torch.arange(N_x) - (N_x//2)
    y = torch.arange(N_y) - (N_y//2)
    X, Y = torch.meshgrid(x,y, indexing='ij')
    zs = origin + step_size * X[..., None] * u + step_size * Y[..., None] * v
    return zs
