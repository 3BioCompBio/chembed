from typing import Dict, List, Optional, Tuple, Sequence

import argparse
import os
from pathlib import Path
import pytorch_lightning as pl

import torch
from torch import Tensor

from chembed import transformer_vae
from chembed.regressor import PropertiesRegressor
from chembed.data_handler import SELFIESDataModule, load_vocab, load_properties_statistics, logits_to_tokens, pad_logits_or_tokens
from chembed.metrics import string_reconstruction_accuracy, token_reconstruction_accuracy, reconstruction_loss, kl_divergence_loss, properties_loss, tanimoto_cos_loss, tanimoto_euclidean_correlation_loss, tanimoto_shifted_euclidean_correlation_loss, tanimoto_shifted_euclidean_correlation_triu_loss 
from chembed.errors import configure_warning_filters
from chembed.utils import set_random_seed_everywhere

class PL_Module(pl.LightningModule):
    def __init__(self, vocab: List[str], vae, train_config: Dict, properties_statistics: Dict[str, Dict[str,float]], regressor: Optional[PropertiesRegressor] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['vae', 'regressor'])
        self.save_hyperparameters({'model_config': vae.model_config})

        self.train_with_properties = train_config['train_with_properties']

        self.vae = vae
        self.regressor = regressor

        self.weight_decay = train_config['weight_decay']
        if train_config['no_weight_decay_for_embeddings']:
            self.special_weight_decay = 0
        else:
            self.special_weight_decay = self.weight_decay

        if 'validate_with_sigma' in train_config:
            self.validate_with_sigma = train_config['validate_with_sigma']
        else:
            self.validate_with_sigma = False

        if 'train_with_greedy_decoding' in train_config:
            self.train_with_greedy_decoding = train_config['train_with_greedy_decoding']
        else:
            self.train_with_greedy_decoding = False

        self.encoder_warmup_steps = train_config['encoder_warmup_steps']
        self.decoder_warmup_steps = train_config['decoder_warmup_steps']
        self.regressor_warmup_steps = train_config['regressor_warmup_steps']
        self.train_decoder_every_n_steps = train_config['train_decoder_every_n_steps']  
        self.train_regressor_every_n_steps = train_config['train_regressor_every_n_steps']  
        self.learning_rate = train_config['learning_rate']

        self.stop_token = self.vae.vocab_to_index['<STOP>']

        self.max_kl_div_coefficient = train_config['kl_div_coefficient']
        self.kl_annealing = train_config['kl_annealing']
        self.kl_cycle_length = train_config['kl_cycle_length']
        self.kl_cycle_R = train_config['kl_cycle_R']
        self.kl_warmup = train_config['kl_warmup']
        self.kl_warmup_steps = train_config['kl_warmup_steps']

        self.properties_loss_coefficient = train_config['properties_loss_coefficient']

        self.train_with_tanimoto_similarity = train_config['train_with_tanimoto_similarity']
        self.tanimoto_loss_coefficient = train_config['tanimoto_loss_coefficient']
        if train_config['tanimoto_loss_type'] == 'cos':
            self.tanimoto_loss = tanimoto_cos_loss 
        elif train_config['tanimoto_loss_type'] == 'euclidean_correlation':
            self.tanimoto_loss = tanimoto_euclidean_correlation_loss 
        elif train_config['tanimoto_loss_type'] == 'shifted_euclidean_correlation':
            self.tanimoto_loss = tanimoto_shifted_euclidean_correlation_loss 
        elif train_config['tanimoto_loss_type'] == 'shifted_euclidean_correlation_triu':
            self.tanimoto_loss = tanimoto_shifted_euclidean_correlation_triu_loss 
        else:
            raise ValueError(train_config['tanimoto_loss_type'])

        if 'debug' in train_config:
            self.debug_mode = train_config['debug']



    def get_kl_div_coefficient(self) -> float:
        if self.kl_annealing:
            t = self.global_step
            C = self.kl_cycle_length
            R = self.kl_cycle_R
            tau = (t % C)/C
            if (tau > R):
                beta = 1
            else:
                beta = tau/R
            return self.max_kl_div_coefficient * beta 
        elif self.kl_warmup:
            t = self.global_step
            N = self.kl_warmup_steps
            if (t > N):
                beta = 1
            else:
                beta = t/N
            return self.max_kl_div_coefficient * beta
        else:
            return self.max_kl_div_coefficient


    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        tokens, target_properties, fingerprints = batch

        vae_output = self.vae(tokens, always_decode_inference=self.train_with_greedy_decoding)

        if self.train_with_greedy_decoding:
            padded_logits, padded_tokens = pad_logits_or_tokens(vae_output['logits'], tokens)
        else:
            padded_logits = vae_output['logits']
            padded_tokens = tokens


        metrics_dict = {}
        metrics_dict['reconstruction_loss'] = reconstruction_loss(padded_logits, padded_tokens)
        metrics_dict['kl_div_loss'] = kl_divergence_loss(vae_output['mu'], vae_output['sigma'])

        kl_coeff = self.get_kl_div_coefficient()
        metrics_dict['kl_div_coefficient'] = kl_coeff
        metrics_dict['loss'] = metrics_dict['reconstruction_loss'] + kl_coeff * metrics_dict['kl_div_loss']

        if self.train_with_properties:  
            predicted_properties = self.regressor(vae_output['z'])
            metrics_dict['properties_loss'] = properties_loss(predicted_properties, target_properties)
            metrics_dict['loss'] += self.properties_loss_coefficient * metrics_dict['properties_loss']
#           predicted_properties_metrics = {'min': predicted_properties.detach().min(dim=0).values, 'max': predicted_properties.detach().max(dim=0).values, 'mean':  predicted_properties.detach().mean(dim=0),'MSE': MSE_unreduced(predicted_properties, target_properties)/len(predicted_properties)}
#           for prop_metrics_type, prop_metrics_vector in predicted_properties_metrics.items():
#               for property_index, property_name in enumerate(self.regressor.properties):
#                   metrics_dict['{}_{}'.format(property_name, prop_metrics_type)] = prop_metrics_vector[property_index].item()

        if self.train_with_tanimoto_similarity:
            metrics_dict['tanimoto_loss'] = self.tanimoto_loss(vae_output['z'], fingerprints)
            metrics_dict['loss'] += self.tanimoto_loss_coefficient * metrics_dict['tanimoto_loss']

        with torch.no_grad():
            output_tokens = logits_to_tokens(padded_logits)
            metrics_dict['token_reconstruction_accuracy'] = token_reconstruction_accuracy(output_tokens, padded_tokens)
            metrics_dict['string_reconstruction_accuracy'] = string_reconstruction_accuracy(output_tokens, padded_tokens)
        

        metrics_dict['output_logits_min'] = vae_output['logits'].detach().min().item()
        metrics_dict['output_logits_max'] = vae_output['logits'].detach().max().item()
        metrics_dict['output_logits_mean'] = vae_output['logits'].detach().mean().item()

        metrics_dict['z_norm_mean'] = torch.linalg.norm(vae_output['z'].flatten(start_dim=1), dim=1).mean()

        metrics_dict['sigma_mean'] = vae_output['sigma'].mean()
        metrics_dict['sigma_norm_mean'] = torch.linalg.norm(vae_output['sigma'].flatten(start_dim=1), dim=1).mean()
        metrics_dict['mean_nb_sigma_elts_lower_than_0.99'] = torch.sum(vae_output['sigma']<0.99, axis=-1).float().mean()

        metrics_dict['mu_mean'] = vae_output['mu'].mean()
        metrics_dict['mu_norm_mean'] = torch.linalg.norm(vae_output['mu'].flatten(start_dim=1), dim=1).mean()

        for k, v in metrics_dict.items():
            if 'loss' in k:
                self.log('train/{}'.format(k), v.detach(), sync_dist=True)#, on_step=True, on_epoch=False)
            else:
                self.log('train/{}'.format(k), v, sync_dist=True)#, on_step=True, on_epoch=False)

        return metrics_dict['loss']
    

    @torch.no_grad()
    def validation_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        tokens, target_properties, fingerprints = batch

        vae_output = self.vae(tokens, always_use_sigma=self.validate_with_sigma)

        padded_logits, padded_tokens = pad_logits_or_tokens(vae_output['logits'], tokens)
        padded_output_tokens = logits_to_tokens(padded_logits)

        metrics_dict = {}
        metrics_dict['reconstruction_loss'] = reconstruction_loss(padded_logits, padded_tokens)
        metrics_dict['kl_div_loss'] = kl_divergence_loss(vae_output['mu'], vae_output['sigma'])
        kl_coeff = self.max_kl_div_coefficient
        metrics_dict['loss'] = metrics_dict['reconstruction_loss'] + kl_coeff * metrics_dict['kl_div_loss']

        if self.train_with_properties:  
            predicted_properties = self.regressor(vae_output['z'])
            metrics_dict['properties_loss'] = properties_loss(predicted_properties, target_properties)
            metrics_dict['loss'] += self.properties_loss_coefficient * metrics_dict['properties_loss']
#           predicted_properties_metrics = {'min': predicted_properties.detach().min(dim=0).values, 'max': predicted_properties.detach().max(dim=0).values, 'mean':  predicted_properties.detach().mean(dim=0),'MSE': MSE_unreduced(predicted_properties, target_properties)/len(predicted_properties)}
#           for prop_metrics_type, prop_metrics_vector in predicted_properties_metrics.items():
#               for property_index, property_name in enumerate(self.regressor.properties):
#                   metrics_dict['{}_{}'.format(property_name, prop_metrics_type)] = prop_metrics_vector[property_index].item()

        if self.train_with_tanimoto_similarity:
            metrics_dict['tanimoto_loss'] = self.tanimoto_loss(vae_output['z'], fingerprints)
            metrics_dict['loss'] += self.tanimoto_loss_coefficient * metrics_dict['tanimoto_loss']

        metrics_dict['token_reconstruction_accuracy'] = token_reconstruction_accuracy(padded_output_tokens, padded_tokens)
        metrics_dict['string_reconstruction_accuracy'] = string_reconstruction_accuracy(padded_output_tokens, padded_tokens)

        metrics_dict['output_logits_min'] = vae_output['logits'].detach().min().item()
        metrics_dict['output_logits_max'] = vae_output['logits'].detach().max().item()
        metrics_dict['output_logits_mean'] = vae_output['logits'].detach().mean().item()

        metrics_dict['z_norm_mean'] = torch.linalg.norm(vae_output['z'].flatten(start_dim=1), dim=1).mean()


        if self.debug_mode:
            logits_train = self.vae.decode_training(vae_output['z'], tokens)
            metrics_dict['debug/train_like_reconstruction_loss'] = reconstruction_loss(logits_train, tokens)
            metrics_dict['debug/train_like_loss'] = metrics_dict['debug/train_like_reconstruction_loss'] + self.get_kl_div_coefficient() * metrics_dict['kl_div_loss']
            if self.train_with_properties:
                metrics_dict['debug/train_like_loss'] += metrics_dict['properties_loss'] 
            output_tokens_train = logits_to_tokens(logits_train)
            metrics_dict['debug/train_like_token_reconstruction_accuracy'] = token_reconstruction_accuracy(output_tokens_train, tokens)
            metrics_dict['debug/train_like_string_reconstruction_accuracy'] = string_reconstruction_accuracy(output_tokens_train, tokens) 

            metrics_dict['debug/train_like_output_logits_min'] = logits_train.detach().min().item()
            metrics_dict['debug/train_like_output_logits_max'] = logits_train.detach().max().item()
            metrics_dict['debug/train_like_output_logits_mean'] = logits_train.detach().mean().item()


        for k, v in metrics_dict.items():
            if 'loss' in k:
                self.log('validation/{}'.format(k), v.detach(), sync_dist=True)#, on_step=True, on_epoch=False)
            else:
                self.log('validation/{}'.format(k), v, sync_dist=True)#, on_step=True, on_epoch=False)


        return metrics_dict['loss']



    def encoder_lr_scheduler(self, step: int) -> float:
        if self.encoder_warmup_steps==0:
            return 1.0
        else:
            return min(step/self.encoder_warmup_steps, 1.0)
    
    def decoder_lr_scheduler(self, step: int) -> float:
        if step < self.encoder_warmup_steps:
            return 0.0
        elif (step - self.encoder_warmup_steps*1) % self.train_decoder_every_n_steps==0:
            if self.decoder_warmup_steps==0:
                return 1.0
            else:
                return min((step - self.encoder_warmup_steps) / (self.decoder_warmup_steps * self.train_decoder_every_n_steps), 1.0)
        else:
            return 0.0

    def regressor_lr_scheduler(self, step: int) -> float:
        if step < self.encoder_warmup_steps:
            return 0.0
        elif (step - self.encoder_warmup_steps*1) % self.train_regressor_every_n_steps==0:
            if self.regressor_warmup_steps==0:
                return 1.0
            else:
                return min((step - self.encoder_warmup_steps) / (self.regressor_warmup_steps * self.train_regressor_every_n_steps), 1.0)
        else:
            return 0.0


    def configure_optimizers(self) -> Dict:
        #params_dict = {'standard_weight_decay':[], 'special_weight_decay':[]}
        model_names = ['encoder', 'decoder']
        if self.train_with_properties:
            model_names.append('regressor')

        params_dict = {'{}_{}'.format(model_name, s):[] for s in ['special', 'standard'] for model_name in model_names}

        # set different optimizers for embedding layers
        for param_name, param in self.named_parameters():
            found=False
            for model_name in model_names:
                if model_name in param_name:
                    found=True
                    if 'embedding' in param_name:
                        params_dict['{}_special'.format(model_name)].append(param)
                    else:
                        params_dict['{}_standard'.format(model_name)].append(param)
            if not found:
                raise Exception(param_name)


        optimizer_params = []
        lr_schedulers = []
        for key, d in params_dict.items():
            if 'special' in key:
                wd = self.special_weight_decay
            elif 'standard' in key:
                wd = self.weight_decay
            else:
                raise Exception(key)
            optimizer_params.append({'params':d, 'lr': self.learning_rate, 'weight_decay':wd})
            if 'encoder' in key:
                lr_schedulers.append(self.encoder_lr_scheduler)
            elif 'decoder' in key:
                lr_schedulers.append(self.decoder_lr_scheduler)
            elif 'regressor' in key:
                lr_schedulers.append(self.regressor_lr_scheduler)
            else:
                raise Exception(key)

        optimizer = torch.optim.Adam(optimizer_params)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedulers)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler':lr_scheduler, 'interval':'step', 'frequency':1}}


def train_model(
        train_path: Path,
        validation_path: Path,
        properties_statistics_path: Path,
        model_config: Dict,
        train_config: Dict,
        log_dir: Path,
        model_name: str,
        devices: List[int],
        vocab: List[str],
        resume_path: Optional[Path] = None,
        use_profiler: bool = False,
        strictly_deterministic_training = False
        ) -> None:

    torch.set_float32_matmul_precision(train_config['float32_matmul_precision'])

    if train_config['train_with_properties']:
        properties_statistics = load_properties_statistics(properties_statistics_path)
        properties = model_config['properties']
    else:
        properties_statistics = None
        properties = []

    data_module = SELFIESDataModule(train_path, validation_path, properties, vocab, train_config)


    vae_cls = getattr(transformer_vae, model_config['vae_model_class'])
    vae = vae_cls(vocab, model_config)

    if train_config['train_with_properties']:
        regressor = PropertiesRegressor(model_config['properties'], properties_statistics, model_config['d_latent'], model_config['layer_norm_in_regressor'], model_config['layer_norm_eps'])
    else:
        regressor = None

    pl_module = PL_Module(vocab, vae, train_config, properties_statistics, regressor=regressor)
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=model_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=train_config['checkpoint_every_n_train_steps'],
                                            save_top_k=train_config['checkpoint_save_top_k'],
                                            save_last=True,
                                            )
    callbacks = [checkpoint_callback, pl.callbacks.LearningRateMonitor()]

    # if dataset is larger than 1 million compounds, perform validation step every 1 million
    val_check_interval = min(1000000/data_module.train.__len__(), 1.0)

    if use_profiler:
        profiler = pl.profilers.PyTorchProfiler(filename='pytorch_profiler.txt',
            profile_memory=True,
            record_shapes=True,
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=False,
            with_modules=True
            ) 
    else:
        profiler = None

    trainer = pl.Trainer(
        callbacks = callbacks,
        logger = logger,
        log_every_n_steps = train_config['log_every_n_steps'],
        max_epochs = train_config['max_epochs'],
        gradient_clip_algorithm = 'norm',
        gradient_clip_val = train_config['gradient_clip_val'],
        devices = devices,
        val_check_interval = val_check_interval,
        deterministic = strictly_deterministic_training,
        profiler = profiler
        )
    
    trainer.fit(pl_module, datamodule = data_module, ckpt_path = resume_path)



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint from which the training will be resumed (if not provided, the training will start from scratch)")
    parser.add_argument("--train_path", type=Path, required=True, help="Path to train dataset (in csv, parquet or pickle format)")
    parser.add_argument("--validation_path", type=Path, required=True, help="Path to train dataset (in csv, parquet or pickle format)")
    parser.add_argument("--properties_statistics_path", type=Path, default=None, help="Path to json file where properties statistics are stored, used when training with an auxiliary properties regression loss. Generated when preprocessing dataset with scripts/preprocess_properties.py")
    parser.add_argument("--properties", nargs='+', default=['MolWt', 'MolLogP', 'TPSA', 'BertzCT', 'Kappa1', 'Kappa2_clipped', 'Kappa3_clipped'], help="List of properties to be considered in the properties regression loss. If the property was clipped, use {property}_clipped.")
    parser.add_argument("--log_dir", type=Path, required=True, help="Path to log directory (to store checkpoints and logs)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name. Logs will be stored in {log_dir}/{model_name}")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of workers for the dataloaders")
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help="List of GPU devices to use")
    parser.add_argument("--checkpoint_every_n_train_steps", type=int, default=1000, help="Save checkpoint every N train steps")
    parser.add_argument("--checkpoint_save_top_k", type=int, default=1, help="Number of last checkpoints to keep")
    parser.add_argument("--log_every_n_steps", type=int, default=1, help="Log every N steps")
    parser.add_argument("--max_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--float32_matmul_precision", type=str, default='medium', help="Controls the precision of matrix multiplications with float32 inputs", choices=['highest', 'high', 'medium'])
    parser.add_argument("--kl_div_coefficient", type=float, default=0.2, help="KL divergence loss coefficient")
    parser.add_argument("--properties_loss_coefficient", type=float, default=1.0, help="Properties regression loss coefficient")
    tanimoto_similarity_group = parser.add_mutually_exclusive_group()
    tanimoto_similarity_group.add_argument("--train_with_tanimoto_similarity", dest="train_with_tanimoto_similarity", action='store_true', help="Train with Tanimoto loss (default)")
    tanimoto_similarity_group.add_argument("--dont_train_with_tanimoto_similarity", dest="train_with_tanimoto_similarity", action='store_false', help="Don't train with Tanimoto loss (default: do)")
    parser.set_defaults(train_with_tanimoto_similarity=True)
    parser.add_argument("--use_precomputed_fingerprints", action='store_true', default=False, help="Use fingerprints precomputed in 'fingerprint' column of the dataset when computing the Tanimoto loss")
    parser.add_argument("--tanimoto_loss_coefficient", type=float, default=0.5, help="Tanimoto loss coefficient")
    parser.add_argument("--tanimoto_loss_type", type=str, default='shifted_euclidean_correlation_triu', choices=['cos', 'euclidean_correlation', 'shifted_euclidean_correlation', 'shifted_euclidean_correlation_triu'], help="Tanimoto loss type (deprecated)")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-2, help="Epsilon parameter for the layer normalization")
    parser.add_argument("--dont_train_with_properties", action='store_true', default=False, help="Don't train with auxiliary properties regression loss (default: do)")
    parser.add_argument("--nb_transformer_layers", type=int, default=6, help="Number of Transformer layers in encoder and in decoder")
    parser.add_argument("--nb_transformer_heads", type=int, default=8, help="Number of attention heads in each Transformer layer")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--layer_norm_in_regressor", action='store_true', default=False, help="Use layer normalization in regressor (default: don't)")
    parser.add_argument("--dim_feedforward_encoder", type=int, default=1024, help="Transformer feedforward dimension in the encoder")
    parser.add_argument("--dim_feedforward_decoder", type=int, default=512, help="Transformer feedforward dimension in the decoder")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--encoder_warmup_steps", type=int, default=100, help="Number of warmup steps for the encoder")
    parser.add_argument("--decoder_warmup_steps", type=int, default=100, help="Number of warmup steps for the decoder")
    parser.add_argument("--regressor_warmup_steps", type=int, default=100, help="Number of warmup steps for the regressor")
    parser.add_argument("--train_decoder_every_n_steps", type=int, default=5, help="Update decoder parameters only every N steps")
    parser.add_argument("--train_regressor_every_n_steps", type=int, default=5, help="Update regressor parameters only every N steps")
    parser.add_argument("--no_log_sigma", action='store_true', default=False, help="Deprecated")
    embeddings_init_group = parser.add_mutually_exclusive_group()
    embeddings_init_group.add_argument("--initialize_embedding_weights", dest='initialize_embedding_weights', action='store_true', help="Initialize token embeddings according to semantic distances (default)")
    embeddings_init_group.add_argument("--dont_initialize_embedding_weights", dest='initialize_embedding_weights', action='store_false', help="Don't initialize token embeddings according to semantic distances (default: do)")
    parser.set_defaults(initialize_embedding_weights=True)
    parser.add_argument("--use_weight_decay_on_embeddings", action='store_true', default=False, help="Apply weight decay to embeddings (default: don't)")
    parser.add_argument("--replace_if_not_in_vocab", action='store_true', default=False, help="Replace token by closest token if not in vocab (default: don't => throw an error if a token in the train set is not in the vocabulary)")
    parser.add_argument("--vocab", type=Path, default=None, help="Path to vocabulary in json format. If not provided, the default vocabulary (chembed/resources/vocab.json) will be used.")
    parser.add_argument("--use_profiler", action='store_true', default=False, help="Use PyTorch profiler")
    parser.add_argument("--kl_annealing", action='store_true', default=False, help="Use KL annealing (default: don't)")
    parser.add_argument("--kl_cycle_length", type=int, default=100000, help="KL annealing cycle length")
    parser.add_argument("--kl_cycle_R", type=float, default=0.5, help="KL annealing R parameter")
    parser.add_argument("--kl_warmup", action='store_true', default=False, help="Use KL warmup strategy (default: don't)")
    parser.add_argument("--kl_warmup_steps", type=int, default=1000000, help="Number of steps for the KL warmup strategy")
    parser.add_argument("--train_with_greedy_decoding", action='store_true', default=False, help="Train with greedy decoding rather than teacher forcing (default: don't)")
    parser.add_argument("--validate_with_sigma", action='store_true', default=False, help="Use sigma in validation metrics (for debug purposes)")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum SELFIES length")
    parser.add_argument("--debug", action='store_true', default=False, help="Set debug mode (log more things)")
    parser.add_argument("--warning_filter", type=str, default='default', help="Set warning verbosity")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--strictly_deterministic", action='store_true', default=False, help="Set torch to be strictly deterministic, requires CUBLAS_WORKSPACE_CONFIG to be set (to :4096:8 (recommended) or :16:8 (limited memory))")
    args = parser.parse_args(argv)

    configure_warning_filters(args.warning_filter)

    set_random_seed_everywhere(args.random_seed)

    model_config = {
        'vae_model_class': 'SELFIESTransformerVAE',
        'max_len': args.max_len,
        'dim_feedforward_encoder': args.dim_feedforward_encoder,
        'nb_layers_encoder': args.nb_transformer_layers,
        'nhead': args.nb_transformer_heads,
        'dim_feedforward_decoder': args.dim_feedforward_decoder,
        'nb_layers_decoder': args.nb_transformer_layers,
        'dropout': args.dropout,
        'min_vae_sigma': 1e-4,
        'properties': args.properties,
        'layer_norm_eps': args.layer_norm_eps,
        'layer_norm_in_regressor': args.layer_norm_in_regressor,
        'use_log_sigma': not args.no_log_sigma,
        'initialize_embedding_weights': args.initialize_embedding_weights,
        'd_model': args.d_model,
        'd_bottleneck': 1,
        'd_latent': args.d_model
        }
    

    if args.kl_annealing and args.kl_warmup:
        raise ValueError("Cannot use KL annealing and KL warmup at the same time")


    train_config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'checkpoint_every_n_train_steps': args.checkpoint_every_n_train_steps,
        'checkpoint_save_top_k': args.checkpoint_save_top_k,
        'log_every_n_steps': args.log_every_n_steps,
        'max_epochs': args.max_epochs,
        'gradient_clip_val': 1.0,
        'weight_decay': args.weight_decay,
        'no_weight_decay_for_embeddings': not args.use_weight_decay_on_embeddings,
        'encoder_warmup_steps': args.encoder_warmup_steps,
        'decoder_warmup_steps': args.decoder_warmup_steps,
        'regressor_warmup_steps': args.regressor_warmup_steps,
        'train_decoder_every_n_steps': args.train_decoder_every_n_steps,
        'train_regressor_every_n_steps': args.train_regressor_every_n_steps,
        'learning_rate': args.learning_rate,
        'float32_matmul_precision': args.float32_matmul_precision,
        'properties_loss_coefficient': args.properties_loss_coefficient,
        'kl_div_coefficient': args.kl_div_coefficient,
        'train_with_tanimoto_similarity': args.train_with_tanimoto_similarity,
        'tanimoto_loss_coefficient': args.tanimoto_loss_coefficient,
        'tanimoto_loss_type': args.tanimoto_loss_type,
        'train_with_properties': not args.dont_train_with_properties,
        'use_normalized_properties': True,
        'use_precomputed_fingerprints': args.use_precomputed_fingerprints,
        'replace_if_not_in_vocab': args.replace_if_not_in_vocab,
        'kl_annealing': args.kl_annealing,
        'kl_cycle_length': args.kl_cycle_length,
        'kl_cycle_R': args.kl_cycle_R,
        'kl_warmup': args.kl_warmup,
        'kl_warmup_steps': args.kl_warmup_steps,
        'train_with_greedy_decoding': args.train_with_greedy_decoding,
        'validate_with_sigma': args.validate_with_sigma,
        'debug': args.debug
        }
    
    if (train_config['train_with_properties']) and (args.properties_statistics_path is None):
        raise Exception("When training with properties, properties statistics should be provided with --properties_statistics_path")

    if args.strictly_deterministic and ("CUBLAS_WORKSPACE_CONFIG" not in os.environ):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG should be set to :4096:8 (recommended) or :16:8 (limited memory) when running with --strictly_deterministic")


    vocab = load_vocab(args.vocab)
    train_model(
            args.train_path,
            args.validation_path,
            args.properties_statistics_path,
            model_config,
            train_config,
            args.log_dir,
            args.model_name,
            args.gpu_devices,
            vocab=vocab,
            resume_path=args.resume,
            use_profiler=args.use_profiler,
            strictly_deterministic_training=args.strictly_deterministic
            )

    return 0


if __name__=="__main__":
    raise SystemExit(main())
