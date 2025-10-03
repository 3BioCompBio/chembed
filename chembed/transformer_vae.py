import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from chembed import data_handler
from chembed.modules import PositionalEncoding
from chembed.token_embeddings_utils import get_initial_embeddings, new_embedding_layer_from_old
from chembed.errors import warn


class SELFIESTransformerVAE(pl.LightningModule):
    def __init__(self, vocab, model_config):
        super().__init__()

        self.model_config = model_config
        
        self.d_model = model_config['d_model'] 

        self.vocab = vocab
        self.nb_tokens = len(self.vocab)
        self.vocab_to_index = {x:i for i, x in enumerate(self.vocab)}
        self.stop_token_index = self.vocab_to_index['<STOP>']
        self.start_token_index = self.vocab_to_index['<START>']
        self.max_len = model_config['max_len']

        self.d_bottleneck = model_config['d_bottleneck']

        self.d_encoder = 2*self.d_model
        self.d_decoder = self.d_model

        self.latent_dimension = (self.d_bottleneck, self.d_model)

        # encoder
        self.encoder_embedding = nn.Sequential(
            nn.Embedding(num_embeddings = self.nb_tokens, embedding_dim = self.d_encoder, scale_grad_by_freq=True),
            PositionalEncoding(self.d_encoder, dropout = model_config['dropout'], max_len = 5000)
            )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = self.d_encoder,
                nhead = model_config['nhead'],
                dim_feedforward = model_config['dim_feedforward_encoder'],
                dropout = model_config['dropout'],
                batch_first = True,
                layer_norm_eps = model_config['layer_norm_eps']
            ),
            num_layers = model_config['nb_layers_encoder']
            )
        self.use_log_sigma = model_config['use_log_sigma']
        self.min_vae_sigma = model_config['min_vae_sigma']

        # decoder
        self.transformer_decoder =  nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model = self.d_decoder,
                nhead = model_config['nhead'],
                dim_feedforward = model_config['dim_feedforward_decoder'],
                dropout = model_config['dropout'],
                batch_first = True,
                layer_norm_eps = model_config['layer_norm_eps']
            ),
            num_layers = model_config['nb_layers_decoder']
        )

        self.decoder_embedding = nn.Sequential(
            nn.Embedding(num_embeddings = self.nb_tokens, embedding_dim = self.d_decoder, scale_grad_by_freq=True),
            PositionalEncoding(self.d_decoder, dropout = model_config['dropout'], max_len = 5000)
            )

        if model_config['initialize_embedding_weights']:
            self.initialize_embedding_weights()

    
    def initialize_embedding_weights(self):
        initial_embeddings_encoder = get_initial_embeddings(self.vocab, self.d_encoder)
        self.encoder_embedding[0].weight.data.copy_(initial_embeddings_encoder)
        initial_embeddings_decoder = get_initial_embeddings(self.vocab, self.d_decoder)
        self.decoder_embedding[0].weight.data.copy_(initial_embeddings_decoder)
    

    @classmethod
    def from_checkpoint(cls, path_to_checkpoint, device, strict=True):
        checkdict = torch.load(path_to_checkpoint, map_location=device)
        if 'model_config' in checkdict['hyper_parameters']:
            model_config = checkdict['hyper_parameters']['model_config']
        else:
            warn(UserWarning, "did not find model_config in hyperparameters, will guess it...")
            nb_layers = max([int(k[len('vae.transformer_decoder.layers.')]) for k in checkdict['state_dict'].keys() if k.startswith('vae.transformer_decoder.layers.')])+1
            model_config = {
            'vae_model_class': 'SELFIESTransformerVAE',
            'max_len': 256,
            'dim_feedforward_encoder': checkdict['state_dict']['vae.transformer_encoder.layers.0.linear1.weight'].shape[0],
            'nb_layers_encoder': nb_layers,
            'nhead': 8,
            'dim_feedforward_decoder': checkdict['state_dict']['vae.transformer_decoder.layers.0.linear1.weight'].shape[0],
            'nb_layers_decoder': nb_layers,
            'dropout': 0.1,
            'min_vae_sigma': 1e-4,
            'properties': list(checkdict['hyper_parameters']['properties_statistics'].keys()),
            'layer_norm_eps': 1e-2,
            'layer_norm_in_regressor': False,
            'use_log_sigma': True,
            'initialize_embedding_weights': False,
            'd_model': checkdict['state_dict']['vae.transformer_decoder.layers.0.linear1.weight'].shape[1],
            'd_bottleneck': 1
            }
        if 'use_log_sigma' not in model_config:
            model_config['use_log_sigma'] = False
        model_config['initialize_embedding_weights'] = False
        vae = cls(vocab=checkdict['hyper_parameters']['vocab'], model_config=model_config)
        vae.load_state_dict({k[len('vae.'):]:v for k, v in checkdict['state_dict'].items() if not k.startswith('regressor')}, strict=strict)
        vae.to(device)
        return vae

    def generate_padding_mask(self, tokens):
        # ignore all padding <STOP> tokens except the first one
        mask = (tokens == self.stop_token_index) 
        inds = mask.float().argmax(dim=-1)
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask


    def encode(self, tokens):
        embedding = self.encoder_embedding(tokens)
        padding_mask = self.generate_padding_mask(tokens)
        encoded = self.transformer_encoder(embedding, src_key_padding_mask = padding_mask)

        split_mu = encoded[..., :self.d_model]
        if self.use_log_sigma:
            split_log_sigma = encoded[..., self.d_model:]
            split_sigma = split_log_sigma.exp() + self.min_vae_sigma 
        else:
            split_sigma = F.softplus(encoded[..., self.d_model:]) + self.min_vae_sigma

        mu = split_mu[:, :self.d_bottleneck, :]
        sigma = split_sigma[:, :self.d_bottleneck, :]

        return mu, sigma
    

    @torch.no_grad()
    def decode_inference(self, z, check_every=32):
        B, _, _ = z.shape
        tokens = torch.full((B, 1), self.start_token_index, device=self.device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        if not hasattr(self, "_tgt_mask"):
            self._tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_len).to(self.device)

        for t in range(self.max_len - 1):
            L = tokens.size(1)
            tgt = self.decoder_embedding(tokens)
            decoded = self.transformer_decoder(tgt=tgt, memory=z, tgt_mask=self._tgt_mask[:L, :L])
            logits = decoded @ self.decoder_embedding[0].weight.T
            next_tokens = data_handler.logits_to_tokens(logits)[:, -1]
            next_tokens = torch.where(finished, torch.full_like(next_tokens, self.stop_token_index), next_tokens)
            tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)
            finished |= next_tokens.eq(self.stop_token_index)

            if (t + 1) % check_every == 0 and finished.all().item():
                break

        return logits

    

    def decode_training(self, z, target_tokens):
        tgt_tokens_shifted = torch.cat([torch.full((target_tokens.shape[0],1), self.start_token_index).to(self.device), target_tokens[:,:-1]], dim=1)
        tgt = self.decoder_embedding(tgt_tokens_shifted)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)
        decoded = self.transformer_decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)
        logits = decoded @ self.decoder_embedding[0].weight.T
        return logits
    

    def forward(self, tokens, always_use_sigma=False, always_decode_inference=False):
        mu, sigma = self.encode(tokens)
        if self.training or always_use_sigma:
            z = mu + sigma * torch.randn_like(mu)
        else:
            z = mu
        if self.training and not always_decode_inference:
            logits = self.decode_training(z, tokens)
        else:
            logits = self.decode_inference(z)
        return {'z':z, 'logits':logits, 'mu':mu, 'sigma':sigma}
    


    def expand_vocab(self, new_vocab):
        old_vocab = self.vocab
        additional_vocab = list(set(new_vocab).difference(old_vocab))

        if len(additional_vocab) > 0:
            self.vocab = old_vocab + additional_vocab 
            self.nb_tokens = len(self.vocab)
            self.vocab_to_index = {x:i for i, x in enumerate(self.vocab)}
            self.stop_token_index = self.vocab_to_index['<STOP>']
            self.start_token_index = self.vocab_to_index['<START>']

            self.encoder_embedding[0] = new_embedding_layer_from_old(self.encoder_embedding[0], additional_vocab, old_vocab)
            self.decoder_embedding[0] = new_embedding_layer_from_old(self.decoder_embedding[0], additional_vocab, old_vocab)


    def set_max_len(self, max_len):
        if max_len != self.max_len:
            self.max_len = max_len
            self.model_config['max_len'] = max_len
