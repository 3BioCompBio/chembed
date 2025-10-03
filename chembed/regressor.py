from typing import List, Dict, Union

from pathlib import Path

import torch
from torch import nn
from torch import Tensor
import pytorch_lightning as pl

class PropertiesRegressor(pl.LightningModule):
    def __init__(self, properties: List[str], properties_statistics: Dict[str, Dict[str,float]], z_dim: int, use_layer_norm: bool, layer_norm_eps: float):
        super().__init__()
        if use_layer_norm:
            self.regressor = nn.Sequential(
                nn.LayerNorm(normalized_shape=z_dim, eps=layer_norm_eps),
                nn.Linear(z_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, len(properties_statistics))
                )
        else:
            self.regressor = nn.Sequential(
                nn.Linear(z_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, len(properties_statistics))
                )
        self.properties = properties
        self.means = torch.tensor([properties_statistics[prop]['mean'] for prop in self.properties])
        self.stds = torch.tensor([properties_statistics[prop]['std'] for prop in self.properties])
    

    @classmethod
    def from_checkpoint(cls, path_to_checkpoint: Path, device: Union[torch.device, str]):
        checkdict = torch.load(path_to_checkpoint, map_location=device)
        if 'model_config' not in checkdict['hyper_parameters']:
            z_dim = 256
            use_layer_norm = False
            layer_norm_eps = None
            properties_statistics = checkdict['hyper_parameters']['properties_statistics']
            properties = list(properties_statistics.keys())
        else:
            if 'd_latent' in checkdict['hyper_parameters']['model_config']: 
                z_dim = checkdict['hyper_parameters']['model_config']['d_latent']
            else:
                z_dim = checkdict['hyper_parameters']['model_config']['d_model']*checkdict['hyper_parameters']['model_config']['d_bottleneck']
            properties_statistics = checkdict['hyper_parameters']['properties_statistics']
            properties = checkdict['hyper_parameters']['model_config']['properties']
            if 'layer_norm_in_regressor' in checkdict['hyper_parameters']['model_config']:
                use_layer_norm = checkdict['hyper_parameters']['model_config']['layer_norm_in_regressor']
                layer_norm_eps = checkdict['hyper_parameters']['model_config']['layer_norm_eps']
            else:
                use_layer_norm = False
                layer_norm_eps = None
        regressor = cls(properties, properties_statistics, z_dim, use_layer_norm, layer_norm_eps)
        regressor.load_state_dict({k[len('regressor.'):]:v for k, v in checkdict['state_dict'].items() if k.startswith('regressor')})
        regressor.to(device)
        return regressor

    def predict_normalized(self, z: Tensor) -> Tensor:
        return self.regressor(z.flatten(start_dim=1))
    
    def denormalize_properties(self, predictions_normalized: Tensor) -> Tensor:
        return self.means.unsqueeze(0).to(self.device) + predictions_normalized*self.stds.unsqueeze(0).to(self.device)
    
    def predict_unnormalized(self, z: Tensor) -> Tensor:
        return self.denormalize_properties(self.predict_normalized(z))
    
    def forward(self, z: Tensor) -> Tensor:
        return self.predict_normalized(z)
