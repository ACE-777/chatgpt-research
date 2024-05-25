import wikipediaapi  # type: ignore
import json
import argparse
from typing import List, Iterable
import numpy as np
import urllib.parse
from unidecode import unidecode

from src import EmbeddingsBuilder, Roberta, Config, Wiki, Index
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from progress.bar import ChargingBar  # type: ignore


def estimate_centroid(data: Iterable[str], tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    embedding_builder = EmbeddingsBuilder(tokenizer, model, False, None)
    embedding_builder.suppress_progress = True
    embeddings = np.empty((0, embedding_builder.embedding_length))
    for page in ChargingBar('Processing texts').iter(data):
        embeddings = np.concatenate([embeddings, embedding_builder.from_text(page)])
    return embeddings.mean(0)


def main(user_input: str) -> str:
    # get wiki text
    wiki_dict = dict()
    words_prob = user_input.split()
    words = [unidecode(urllib.parse.unquote(item)) for item in words_prob]
    for page in words:
        wiki_dict |= Wiki.parse(page)
    keys = list(wiki_dict.keys())
    keys_update = list()

    for key in keys:
        if "#" not in key:
            key = key + "#"
            keys_update.append(key)
            continue

        keys_update.append(key)

    values = [wiki_dict[key] for key in keys_update]
    key_value_pairs = [{key: value} for key, value in zip(keys_update, values)]
    with open("./artifacts/scrape_wiki.json", "w") as json_file:
        json.dump(key_value_pairs, json_file, indent=4)

    # calculate centroid
    roberta = Roberta.get_default()
    page_names = words
    texts: List[str] = []
    for name in ChargingBar('Loading articles').iter(page_names):
        texts += Wiki.parse(name).values()
    centroid = estimate_centroid(texts, *roberta)
    np.save(Config.centroid_file, centroid)

    # build search database
    index = Index.from_config_wiki_input(True, Config.centroid_file, page_names)
    index.save(Config.index_file, Config.mapping_file)

    res = "Success"

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", help="User input value", type=str)
    args = parser.parse_args()

    articles = args.articles

    result = main(articles)

    dictionary = {
        'result': result,
    }

    json_output = json.dumps(dictionary, indent=4)
    print(json_output)
