from typing import List, Dict, SupportsInt, Sequence, Iterable, Tuple, Optional

from pathlib import Path
import pandas as pd
import ast
import selfies as sf

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import pytorch_lightning as pl

from chembed.utils import read_df, read_json, get_resource_file 
from chembed.token_embeddings_utils import get_closest_token
from chembed.errors import warn, UnknownTokenError, TokenReplacementWarning
from chembed.mol_utils import get_fingerprint_from_smiles


START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"
PAD_TOKEN = 1


_PAD_LOGITS_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def load_vocab(vocab_path: Path = None) -> List:
    if vocab_path is None:
        vocab_path = get_resource_file('vocab.json')
    vocab = read_json(vocab_path)
    if (START_TOKEN not in vocab) or (STOP_TOKEN not in vocab):
        vocab = [START_TOKEN, STOP_TOKEN] + vocab
    assert vocab[PAD_TOKEN] == STOP_TOKEN
    return vocab


def load_properties_statistics(input_path: Path) -> Dict:
    return read_json(input_path)


def get_word_to_index(w: str, word_to_index: Dict[str,int], replace_if_not_in_vocab: bool = False) -> int:
    if w in word_to_index:
        return word_to_index.get(w)
    elif replace_if_not_in_vocab:
        if w in ['[2H]', '[3H]']:
            replacement = '[H]'
            raise TokenReplacementWarning(w, replacement)
        else:
            replacement = get_closest_token(w, word_to_index.keys())
        warn(TokenReplacementWarning(w, replacement))
        return word_to_index.get(replacement)
    else:
        raise UnknownTokenError(w)


def selfies_string_to_tokens(selfies_string: str, word_to_index: Dict[str,int], replace_if_not_in_vocab: bool = False) -> Tensor:
    tokens = [get_word_to_index(w, word_to_index, replace_if_not_in_vocab) for w in sf.split_selfies(selfies_string)]
    tokens = tokens + [word_to_index['<STOP>']]
    return torch.tensor(tokens)


def tokens_to_selfies_string(tokens: Iterable[SupportsInt], vocab: List[str]) -> str:
    return ''.join([vocab[i] for i in tokens if (vocab[i]!='<START>') and (vocab[i]!='<STOP>')])


def tokens_to_selfies_strings(batch_tokens: Iterable[Iterable[SupportsInt]], vocab: List[str]) -> List[str]:
    return [tokens_to_selfies_string(ts, vocab) for ts in batch_tokens]


def logits_to_tokens(logits: Tensor) -> Tensor:
    return logits.argmax(dim=-1)


def tokens_to_logits(tokens: Tensor, vocab_size: int, eps: float = 1e-6) -> Tensor:
    out_shape = (*tokens.shape, vocab_size)
    logits = torch.full(out_shape, eps/vocab_size, device=tokens.device)
    logits.scatter_(-1, tokens.unsqueeze(-1), 1.0 - eps)
    return torch.log(logits)


def pad_logits_to_match_length(logits: Tensor, max_length: int) -> Tensor:
    B, L, N = logits.shape

    if max_length <= L:
        return logits

    device = logits.device
    key = (L, N, device)

    pad_vec = _PAD_LOGITS_CACHE.get(key) # cache padding vectors to save time

    if pad_vec is None:
        # build a proba vector p[PAD_TOKEN] = 1-eps and p[i!=PAD_TOKEN]=eps/(N-1)
        eps = 1e-6
        p = torch.full((N,), eps/N, device=device, dtype=logits.dtype)
        p[PAD_TOKEN] = 1.0 - eps
        pad_vec = torch.log(p)
        _PAD_LOGITS_CACHE[key] = pad_vec

    padded_logits = logits.new_empty((B, max_length, N))
    padded_logits[:, :L, :] = logits
    padded_logits[:, L:, :] = pad_vec
    return padded_logits
    

def pad_tokens_to_match_length(tokens: Sequence[Tensor], max_length: int) -> Tensor:
    return torch.vstack([F.pad(x, (0, max_length - x.shape[-1]), value=PAD_TOKEN) for x in tokens])


def pad_tokens(batch_tokens: Iterable[Iterable[SupportsInt]]) -> Tensor:
    max_length = max(x.shape[-1] for x in batch_tokens)
    return pad_tokens_to_match_length(batch_tokens, max_length)


def pad_logits_or_tokens(logits: Tensor, tokens: Tensor) -> Tensor:
    logits_length = logits.shape[1]
    tokens_length = tokens.shape[1]
    if tokens_length < logits_length:
        tokens = pad_tokens_to_match_length(tokens, logits_length)
    elif logits_length < tokens_length:
        logits = pad_logits_to_match_length(logits, tokens_length)
    return logits, tokens


def collate_fn_tokens(batch_tokens: Iterable[Iterable[SupportsInt]]) -> Tensor:
    return pad_tokens(batch_tokens)


def collate_fn(data: Sequence[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]) -> Tensor:
    tokens, properties, fingerprints = map(list, zip(*data))
    collate_tokens = collate_fn_tokens(tokens)
    if properties[0] is not None:
        collate_properties = torch.vstack(properties)
    else:
        collate_properties = None
    if fingerprints[0] is not None:
        collate_fingerprints = torch.vstack(fingerprints)
    else:
        collate_fingerprints = None
    return collate_tokens, collate_properties, collate_fingerprints


class SELFIESDataset(Dataset):
    def __init__(
                self, df: pd.DataFrame,
                vocab: List[str],
                properties: List[str],
                selfies_column_name: str = 'selfies',
                use_normalized_properties: bool = True,
                return_fingerprints: bool = False,
                return_properties: bool = True,
                replace_if_not_in_vocab: bool = False
                ):
        self.df = df
        self.selfies_column_name = selfies_column_name
        self.vocab = vocab
        self.vocab_to_index = {x:i for i, x in enumerate(self.vocab)}
        self.properties = properties
        self.return_properties = return_properties
        self.use_normalized_properties = use_normalized_properties
        self.return_fingerprints = return_fingerprints
        self.replace_if_not_in_vocab = replace_if_not_in_vocab
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = selfies_string_to_tokens(row[self.selfies_column_name], self.vocab_to_index, self.replace_if_not_in_vocab)
        if self.return_properties:
            if self.use_normalized_properties:
                target_properties = torch.tensor([row['normalized_{}'.format(prop_name)] for prop_name in self.properties], dtype=torch.float)
            else:
                target_properties = torch.tensor([row[prop_name] for prop_name in self.properties], dtype=torch.float)
        else:
            target_properties = None
        if self.return_fingerprints:
            if 'fingerprint' in row:
                fp_raw = row['fingerprint']
                if isinstance(fp_raw, str):
                    fp_raw = ast.literal_eval(fp_raw) 
                fingerprint = torch.tensor(fp_raw)
            else:
                fingerprint = torch.tensor(get_fingerprint_from_smiles(row['smiles'])) 
        else:
            fingerprint = None
        return tokens, target_properties, fingerprint


class SELFIESDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_data_path: Path,
            validation_data_path: Path,
            properties: List[str],
            vocab: List[str],
            train_config: Dict
            ):
        super().__init__()
        self.batch_size = train_config['batch_size']
        self.num_workers = train_config['num_workers']
        self.use_fingerprints=train_config['train_with_tanimoto_similarity']
        self.use_properties = train_config['train_with_properties']
        self.use_normalized_properties = train_config['use_normalized_properties']
        if self.use_normalized_properties:
            required_columns = ['selfies'] + ['normalized_{}'.format(prop) for prop in properties]
        else:
            required_columns = ['selfies'] + properties 
        if self.use_fingerprints:
            if train_config['use_precomputed_fingerprints']:
                required_columns += ['fingerprint']
            else:
                required_columns += ['smiles']
        self.replace_if_not_in_vocab = train_config['replace_if_not_in_vocab']
        df_train = read_df(train_data_path, required_columns)
        df_validation = read_df(validation_data_path, required_columns)
        self.train = SELFIESDataset(df_train, vocab, properties, return_fingerprints=self.use_fingerprints, return_properties=self.use_properties, use_normalized_properties=self.use_normalized_properties, replace_if_not_in_vocab=self.replace_if_not_in_vocab)
        self.val = SELFIESDataset(df_validation, vocab, properties, return_fingerprints=self.use_fingerprints, return_properties=self.use_properties, use_normalized_properties=self.use_normalized_properties, replace_if_not_in_vocab=self.replace_if_not_in_vocab)

        if len(self.train) < self.batch_size:
            warn(UserWarning, f"batch_size={self.batch_size} < len(train)={len(self.train)} -> batch_size was set to len(train)")
            self.batch_size = len(self.train)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers, drop_last=True)
