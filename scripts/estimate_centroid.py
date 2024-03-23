from typing import List, Iterable, Set

import numpy as np

from src import EmbeddingsBuilder, Roberta, Config, Wiki
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from progress.bar import ChargingBar  # type: ignore


def estimate_centroid(data: Iterable[str], tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    embedding_builder = EmbeddingsBuilder(tokenizer, model, False, None)
    embedding_builder.suppress_progress = True
    embeddings = np.empty((0, embedding_builder.embedding_length))
    for page in ChargingBar('Processing texts').iter(data):
        embeddings = np.concatenate([embeddings, embedding_builder.from_text(page)])
    return embeddings.mean(0)


def main() -> None:
    roberta = Roberta.get_default()
    page_names = Config.page_names + Config.unrelated_page_names
    texts: List[str] = []
    for name in ChargingBar('Loading articles').iter(page_names):
        texts += Wiki.parse(name).values()
    centroid = estimate_centroid(texts, *roberta)
    np.save(Config.centroid_file, centroid)

    print('Centroid:')
    print(repr(centroid))


if __name__ == '__main__':
    main()
