from typing import List
import math

import torch
from torch import nn
from torch import Tensor

class PositionalEncoding(nn.Module):
        # https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308
    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) 

    def forward(self, x: Tensor) -> Tensor: 
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class FC_Module(nn.Module):
    """ Fully-Connected Layers with ReLU activations and dropout """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        fc_layers = []
        dimensions = [input_dim] + hidden_dims
        for layer_number in range(len(dimensions)-2):
                input_dim = dimensions[layer_number]
                output_dim = dimensions[layer_number+1]
                fc_layers.append(nn.Linear(input_dim, output_dim))
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(dimensions[-2],dimensions[-1]))
        self.fc_module = nn.Sequential(*fc_layers)
        self.output_dim = dimensions[-1]
        
    def forward(self, x: Tensor) -> Tensor:
        return self.fc_module(x)
