from typing import Tuple
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
import torch  # type: ignore

from .Config import Config


class Roberta:
    @staticmethod
    def get_default() -> Tuple[RobertaTokenizer, RobertaModel]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t = RobertaTokenizer.from_pretrained(Config.model_name)
        m = RobertaModel.from_pretrained(Config.model_name).to(device)
        return t, m
