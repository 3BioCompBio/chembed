from typing import List, Dict, Sequence
from numpy.typing import NDArray

import re
import numpy as np
from sklearn import manifold
from rdkit import Chem

import torch
from torch import nn
from torch import Tensor

PERIODIC_TABLE = Chem.GetPeriodicTable()
HEAVY_ATOMS = [PERIODIC_TABLE.GetElementSymbol(i) for i in range(1, 119)]
HEAVY_ATOMS_ORGANIC = ['B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', 'Si', 'Sn']
ATOMS = HEAVY_ATOMS + ['H']
TOPOLOGY_TOKENS = ['Branch{}'.format(i) for i in range(5)] + ['Ring{}'.format(i) for i in range(5)]
BONDS = ['-/', '-\\', '/', '\\', '=', '#']
STEREOS = ['@', '@@']
SUBTOKENS_LIST = HEAVY_ATOMS + TOPOLOGY_TOKENS
SUBTOKENS_LIST_SORTED = sorted(SUBTOKENS_LIST, key=len, reverse=True)

_CLOSEST_TOKENS_CACHE = {}


def get_main_token_type(main_token: str) -> str:
    if main_token in ATOMS:
        return 'atom'
    elif main_token in TOPOLOGY_TOKENS:
        return 'topology'
    else:
        raise Exception(main_token)


def get_atoms_distance(x1: str, x2: str) -> float:
    """ returns an arbitrary distance between two atom tokens: 0 if atoms are the same, 0.9 if they are different """
    if x1==x2:
        return 0
    else:
        return 0.9

def get_topology_tokens_distance(x1: str, x2: str) -> float:
    """ returns an arbitrary distance between two topology tokens: 0 if atoms are the same, 0.9 if they are from the same topology class, 1 otherwise """
    if x1==x2:
        return 0
    elif len(x1)==len(x2): # both Branch or both Ring
        return 0.9
    else:
        return 1


def get_main_tokens_distance(x1: str, x2: str) -> float:
    if x1==x2:
        return 0
    else:
        x1_type = get_main_token_type(x1)
        x2_type = get_main_token_type(x2)
        if x1_type != x2_type:
            return 1
        else:
            if x1_type=='atom':
                return get_atoms_distance(x1, x2)
            else:
                return get_topology_tokens_distance(x1, x2)


def get_bond_type(bond: str) -> int:
    """ bond str to int """ 
    if bond=='=':
        return 2
    elif bond=='#':
        return 3
    else:
        return 1

def get_bonds_distance(bond1: str, bond2: str) -> float:
    if bond1==bond2:
        return 0
    else:
        bond_type1 = get_bond_type(bond1)
        bond_type2 = get_bond_type(bond2)
        if bond_type1==bond_type2:
            return 0.1
        else:
            return abs(bond_type1-bond_type2)/2

def get_nb_hs(hs: str) -> int:
    """ returns nb hydrogens from the "hydrogen" part of the token """
    if hs is None:
        return None
    elif len(hs)==0:
        return None
    else:
        assert(hs.startswith('H'))
        if hs=='H':
            return 1
        else:
            nb_hs = int(hs[1:])
            return nb_hs

def get_hs_distance(h1: str, h2: str) -> float:
    if h1==h2:
        return 0
    else:
        nbh1 = get_nb_hs(h1)
        nbh2 = get_nb_hs(h2)
        if (nbh1 is None) or (nbh2 is None):
            return 1.0
        else:
            return abs(nbh1-nbh2)/4

def get_charge(charge: str) -> int:
    if charge is None:
        return 0
    elif charge=='':
        return 0
    else:
        return int(charge)

def get_charges_distance(charge1: str, charge2: str) -> float:
    charge1 = get_charge(charge1)
    charge2 = get_charge(charge2)
    return abs(charge1-charge2)/5

def get_stereos_distance(stereo1: str, stereo2: str) -> float:
    if stereo1 == stereo2:
        return 0
    else:
        return 1

DISTANCES_DICT = {
    'bond': {'coeff': 2, 'distance': get_bonds_distance},
    'main_token': {'coeff': 5, 'distance': get_main_tokens_distance},
    'stereo': {'coeff': 1, 'distance': get_stereos_distance},
    'hs': {'coeff': 1, 'distance': get_hs_distance},
    'charge': {'coeff': 1, 'distance': get_charges_distance}
}

TOTAL_DISTANCES_WEIGHTS = sum([v['coeff'] for v in DISTANCES_DICT.values()])

def break_down_token(token: str):
    if token != '[H]':
        pattern = rf"^\[((?:{'|'.join(map(re.escape, BONDS))})?)" \
          rf"({'|'.join(SUBTOKENS_LIST_SORTED)})" \
          rf"((?:{'|'.join(map(re.escape, STEREOS))})?)" \
          rf"(H\d*)?([+-]\d*)?\]$"
        return re.match(pattern, token).groups()
    else:
        return '', 'H', '', None, None

def break_down_token_to_dict(token: str) -> Dict[str,str]:
    res = break_down_token(token)
    return {k:v for k, v in zip(['bond', 'main_token', 'stereo', 'hs', 'charge'], res)}


def get_total_tokens_distance_from_broken_down_tokens(t1: Dict[str,str], t2: Dict[str,str]) -> float:
    distance = 0
    for name, d in DISTANCES_DICT.items():
        distance += d['coeff'] * d['distance'](t1[name], t2[name])
    distance = distance / TOTAL_DISTANCES_WEIGHTS
    return distance

def get_total_tokens_distance(token1: str, token2: str) -> float:
    t1 = break_down_token_to_dict(token1)
    t2 = break_down_token_to_dict(token2)
    return get_total_tokens_distance_from_broken_down_tokens(t1, t2)

def get_selfies_tokens_distance_matrix(selfies_tokens: List[str]) -> NDArray:
    D_tokens = np.zeros((len(selfies_tokens), len(selfies_tokens)))
    for i, token1 in enumerate(selfies_tokens):
        for j in range(i, len(selfies_tokens)):
            token2 = selfies_tokens[j]
            D_tokens[i,j] = get_total_tokens_distance(token1, token2)
            D_tokens[j,i] = D_tokens[i,j]
    return D_tokens


def get_initial_embeddings(vocab, embedding_dim, scaler=40.0):
    assert(vocab[0]=='<START>')
    assert(vocab[1]=='<STOP>')
    selfies_tokens = vocab[2:]
    D_selfies = get_selfies_tokens_distance_matrix(selfies_tokens)
    D_total = np.ones((len(vocab),len(vocab)))
    D_total[2:,2:] = D_selfies
    D_total = scaler * D_total
    mds = manifold.MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=42).fit(D_total)
    return torch.from_numpy(mds.embedding_)


def get_closest_token_from_token_dict(token_broken: Dict, tokens: Sequence[str]) -> str:
    tokens = list(tokens)
    closest_token = tokens[0]
    smallest_distance = np.inf
    for t in tokens:
        if not t.startswith('['):
            d = np.inf
        else:
            try:
                tb = break_down_token_to_dict(t)
                d = get_total_tokens_distance_from_broken_down_tokens(token_broken, tb)
                if d < smallest_distance:
                    closest_token = t
                    smallest_distance = d
            except Exception:
                d = np.inf
    return closest_token



def get_closest_token(token: str, tokens: Sequence[str]) -> str:
    if token in _CLOSEST_TOKENS_CACHE:
        return _CLOSEST_TOKENS_CACHE.get(token)
    else:
        token_broken = break_down_token_to_dict(token)
        if (token_broken['main_token'] not in HEAVY_ATOMS_ORGANIC) and (token_broken['main_token'] not in TOPOLOGY_TOKENS):
            token_broken['main_token'] = 'C'
        replacement = get_closest_token_from_token_dict(token_broken, tokens)
        _CLOSEST_TOKENS_CACHE[token] = replacement
        return replacement


def initialize_additional_embeddings(additional_vocab: List[str], old_vocab: List[str], old_embeddings: Tensor) -> Tensor:
    new_embeddings = []

    assert(len(old_embeddings)==len(old_vocab))
    old_vocab_to_embeddings = {old_vocab[i]: old_embeddings[i] for i in range(len(old_vocab))}

    embedding_shape = old_embeddings[0].shape

    for new_token in additional_vocab:
        try:
            token_broken = break_down_token_to_dict(new_token)
            closest_token = get_closest_token_from_token_dict(token_broken, old_vocab)
            new_embedding = old_vocab_to_embeddings[closest_token]
        except Exception:
            new_embedding = torch.randn(embedding_shape) 
        new_embeddings.append(new_embedding)
    
    return torch.vstack(new_embeddings)



def new_embedding_layer_from_old(old_layer: nn.Embedding, additional_vocab: List[str], old_vocab: List[str]) -> nn.Embedding:
    old_nb_tokens = len(old_vocab)
    nb_tokens = old_nb_tokens + len(additional_vocab)
    d_encoder = old_layer.weight.data[0].shape[-1]
    new_encoder_embedding_layer = nn.Embedding(num_embeddings=nb_tokens, embedding_dim=d_encoder, scale_grad_by_freq=True)
    old_embeddings = old_layer.weight.data
    new_encoder_embedding_layer.weight.data[:old_nb_tokens] = old_embeddings 
    new_encoder_embedding_layer.weight.data[old_nb_tokens:] = initialize_additional_embeddings(additional_vocab, old_vocab, old_embeddings) 
    return new_encoder_embedding_layer
