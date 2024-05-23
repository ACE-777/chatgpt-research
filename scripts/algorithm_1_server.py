import collections
import argparse
import json


from transformers import RobertaTokenizer  # type: ignore

from src import Config, EmbeddingsBuilder, Index,Roberta
from typing import Dict, Any


tokenizer, model = Roberta.get_default()


def build_dict_for_color(links: list[str], uniq_color: int) -> Dict[str, str]:
    filtered_links = [link for link in links if link is not None]
    dictionary_of_links = dict(collections.Counter(filtered_links))
    sorted_dict = dict(sorted(dictionary_of_links.items(), key=lambda x: x[1], reverse=True))
    links_with_uniq_colors = dict(list(sorted_dict.items())[:uniq_color])
    uniq_color_dict = {
        'Fuchsia': 'color1',
        'MediumPurple': 'color2',
        'DarkViolet': 'color3',
        'DarkMagenta': 'color4',
        'Indigo': 'color5'
    }

    for link, (_, color_hex) in zip(links_with_uniq_colors, uniq_color_dict.items()):
        links_with_uniq_colors[link] = color_hex

    return links_with_uniq_colors


def prob_test_wiki_with_colored(index: Index, text: str) -> tuple[Any, list[str], Any]:
    embeddings = EmbeddingsBuilder(tokenizer, model, normalize=True, centroid_file=Config.centroid_file).from_text(text)
    tokens = build_list_of_tokens_input(text)
    result_sources, result_dists = index.get_embeddings_source(embeddings)
    return result_sources, tokens, result_dists


def build_list_of_tokens_input(text: str) -> list[str]:
    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
    tokens = tokenizer.tokenize(text)

    return tokens


def main(user_input: str) -> tuple[Any, list[str], Any]:
    read_index: bool = True

    if read_index:
        index = Index.load(Config.index_file, Config.mapping_file)
    else:
        index = Index.from_embeddings()

    return prob_test_wiki_with_colored(index, user_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--userinput", help="User input value", type=str)
    args = parser.parse_args()

    userinput = args.userinput

    result_sources, tokens, result_dists = main(userinput)

    float_list = result_dists.tolist()
    str_list = result_sources.tolist()

    dictionary = {
        'result_sources': str_list,
        'tokens': tokens,
        'result_dists': float_list,
    }

    json_output = json.dumps(dictionary, indent=4)
    print(json_output)
