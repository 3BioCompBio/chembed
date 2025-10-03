from typing import Optional
import warnings
from sklearn.exceptions import DataConversionWarning
from rdkit import RDLogger

class UnknownTokenError(KeyError):
    def __init__(self, token: str):
        msg = f"Token not found in vocab: {token}. Use replace_if_not_in_vocab option if you want it automatically replaced with the closest token in vocab."
        super().__init__(msg)
        self.token = token

class TokenReplacementWarning(UserWarning):
    def __init__(self, original_token: str, replacement_token: str):
        msg = f"Token not found in vocab: {original_token}. Replacing with {replacement_token}."
        super().__init__(msg)

def warn(warning: Warning, message: Optional[str] = None) -> None:
    if message:
        warnings.warn(message, warning)
    else:
        warnings.warn(warning)

def configure_warning_filters(action: str='default') -> None:
    warnings.simplefilter(action, TokenReplacementWarning)
    warnings.filterwarnings('ignore', message=".*API of nested tensors is in prototype stage.*")
    warnings.filterwarnings('ignore', message=r"You are using `torch\.load` with `weights_only=False`")
    warnings.filterwarnings('ignore', category=DataConversionWarning, message=".*Data was converted to boolean for metric jaccard.*")
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
