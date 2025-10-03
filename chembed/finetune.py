from typing import List, Dict, Optional, Sequence

import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl

from chembed import checkpoint_utils
from chembed.data_handler import SELFIESDataModule, load_vocab, load_properties_statistics
from chembed.train import PL_Module 
from chembed.regressor import PropertiesRegressor
from chembed.errors import configure_warning_filters, warn
from chembed.utils import set_random_seed_everywhere

def finetune_vae(train_path: Path, validation_path: Path, properties_statistics_path: Path, vae, train_config: Dict, log_dir: Path, model_name: str, devices: List[int], vocab: List[str], resume_path: Optional[Path] = None) -> None:

    torch.set_float32_matmul_precision(train_config['float32_matmul_precision'])

    if train_config['train_with_properties']:
        properties_statistics = load_properties_statistics(properties_statistics_path)
        properties = train_config['properties']
        regressor = PropertiesRegressor(train_config['properties'], properties_statistics, vae.d_model, train_config['layer_norm_in_regressor'], train_config['layer_norm_eps'])
    else:
        properties_statistics = None
        properties = []
        regressor = None

    data_module = SELFIESDataModule(train_path, validation_path, properties, vocab, train_config)

    pl_module = PL_Module(vocab, vae, train_config, properties_statistics, regressor=regressor)

    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name=model_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=train_config['checkpoint_every_n_train_steps'],
                                            save_top_k=train_config['checkpoint_save_top_k'],
                                            save_last=True,
                                            )
    callbacks = [checkpoint_callback, pl.callbacks.LearningRateMonitor()]

    # if dataset is larger than 1 million compounds, perform validation step every 1 million
    val_check_interval = min(1000000/data_module.train.__len__(), 1.0)


    trainer = pl.Trainer(
        callbacks = callbacks,
        logger = logger,
        log_every_n_steps=train_config['log_every_n_steps'],
        max_epochs = train_config['max_epochs'],
        gradient_clip_algorithm = 'norm',
        gradient_clip_val = train_config['gradient_clip_val'],
        devices = devices,
        val_check_interval = val_check_interval,
        )
    
    trainer.fit(pl_module, datamodule = data_module, ckpt_path = resume_path)



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint from which the training will be resumed (if not provided, the training will start from scratch)")
    vae_group = parser.add_mutually_exclusive_group()
    vae_group.add_argument("--model", type=str, help="HuggingFace model repo id", default="3BioCompBio/chembed-default")
    vae_group.add_argument("--checkpath", type=Path, help="Path to local checkpoint (if not using HuggingFace)", default=None)
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
    parser.add_argument("--layer_norm_in_regressor", action='store_true', default=False, help="Use layer normalization in regressor (default: don't)")
    properties_group = parser.add_mutually_exclusive_group()
    properties_group.add_argument("--train_with_properties", dest="train_with_properties", action='store_true', help="Finetune with auxiliary property regression task (default: don't)")
    properties_group.add_argument("--dont_train_with_properties", dest="train_with_properties", action='store_false', help="Don't finetune with auxiliary property regression task (default)")
    parser.set_defaults(train_with_properties = False)
    parser.add_argument("--train_decoder_every_n_steps", type=int, default=5, help="Update decoder parameters only every N steps")
    parser.add_argument("--train_regressor_every_n_steps", type=int, default=5, help="Update regressor parameters only every N steps")
    parser.add_argument("--use_weight_decay_on_embeddings", action='store_true', default=False, help="Apply weight decay to embeddings (default: don't)")
    parser.add_argument("--vocab", type=Path, default=None, help="Path to vocabulary in json format. If not provided, the vocabulary from the pretrained model will be used")
    parser.add_argument("--kl_annealing", action='store_true', default=False, help="Use KL annealing (default: don't)")
    parser.add_argument("--kl_cycle_length", type=int, default=100000, help="KL annealing cycle length")
    parser.add_argument("--kl_cycle_R", type=float, default=0.5, help="KL annealing R parameter")
    parser.add_argument("--train_with_greedy_decoding", action='store_true', default=False, help="Train with greedy decoding rather than teacher forcing (default: don't)")
    parser.add_argument("--validate_with_sigma", action='store_true', default=False, help="Use sigma in validation metrics (for debug purposes)")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum SELFIES length")
    parser.add_argument("--debug", action='store_true', default=False, help="Set debug mode (log more things)")
    parser.add_argument("--warning_filter", type=str, default='default', help="Set warning verbosity")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args(argv)

    configure_warning_filters(args.warning_filter)

    set_random_seed_everywhere(args.random_seed)

    if (args.train_with_properties) and (not args.properties_statistics_path):
        raise ValueError("--properties_statistics_path must be provided when --train_with_properties (obtained when you ran scripts/preprocess_properties.py)")

    if (not args.train_with_properties) and (args.properties_statistics_path):
        raise ValueError("You provided --properties_statistics_path but you're training with --dont_train_with_properties")

    if not args.vocab:
        warn(UserWarning, "You did not provide a new vocabulary file with --vocab, so the VAE's vocabulary will be used. If your train file contains SELFIES tokens that are not in its vocabulary, this will crash.")

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
        'encoder_warmup_steps': 0,
        'decoder_warmup_steps': 0,
        'regressor_warmup_steps': 0,
        'train_decoder_every_n_steps': args.train_decoder_every_n_steps,
        'train_regressor_every_n_steps': args.train_regressor_every_n_steps,
        'learning_rate': args.learning_rate,
        'float32_matmul_precision': args.float32_matmul_precision,
        'properties_loss_coefficient': args.properties_loss_coefficient,
        'kl_div_coefficient': args.kl_div_coefficient,
        'train_with_tanimoto_similarity': args.train_with_tanimoto_similarity,
        'tanimoto_loss_coefficient': args.tanimoto_loss_coefficient,
        'tanimoto_loss_type': args.tanimoto_loss_type,
        'train_with_properties': args.train_with_properties,
        'use_normalized_properties': True,
        'use_precomputed_fingerprints': args.use_precomputed_fingerprints,
        'replace_if_not_in_vocab': False,
        'properties': args.properties,
        'layer_norm_eps': args.layer_norm_eps,
        'layer_norm_in_regressor': args.layer_norm_in_regressor,
        'kl_annealing': args.kl_annealing,
        'kl_cycle_length': args.kl_cycle_length,
        'kl_cycle_R': args.kl_cycle_R,
        'kl_warmup': False,
        'kl_warmup_steps': None,
        'train_with_greedy_decoding': args.train_with_greedy_decoding,
        'validate_with_sigma': args.validate_with_sigma,
        'debug': args.debug
        }


    if args.resume:
        vae = checkpoint_utils.load_vae_from_checkpoint(args.resume, device = torch.device('cpu'))

    else:
        if args.checkpath:
            vae = checkpoint_utils.load_vae_from_checkpoint(args.checkpath, 'cpu') 
        else:
            vae = checkpoint_utils.load_vae_from_hub(device = 'cpu', repo_id = args.model)


        vae.set_max_len(args.max_len)
        
        if vae.vocab is not None:
            new_vocab = load_vocab(args.vocab) 
            vae.expand_vocab(new_vocab)

    finetune_vae(args.train_path, args.validation_path, args.properties_statistics_path, vae, train_config, args.log_dir, args.model_name, args.gpu_devices, vocab=vae.vocab, resume_path=args.resume)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
