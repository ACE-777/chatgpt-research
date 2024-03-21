import collections
# import faiss
import argparse
import json


from scripts.model_of_GPT import build_page_template
from scripts.build_index_from_potential_sources import build_index
from transformers import RobertaTokenizer  # type: ignore

from src import SourceMapping, Config, EmbeddingsBuilder, Embeddings, Index,Roberta
from typing import Dict, List, Optional, Tuple, Any



tokenizer, model = Roberta.get_default()

# from 'Childhood in Tupelo' section
childhood_w_refs = (
    "Presley's father Vernon was of German, Scottish, and English origins,[12] and a descendant of the "
    "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood).[8] Presley's mother "
    "Gladys was Scots-Irish with some French Norman ancestry.[13] She and the rest of the family believed "
    "that her great-great-grandmother, Morning Dove White, was Cherokee.[14][15][16] This belief was restated "
    "by Elvis's granddaughter Riley Keough in 2017.[17] Elaine Dundy, in her biography, supports the belief.["
    "18]"
)

childhood_url = "https://en.wikipedia.org/wiki/Elvis_Presley#Childhood_in_Tupelo"


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


def prob_test_wiki_with_colored(index: Index, text: str, expected_url: str,
                                uniq_color: int) -> tuple[Any, list[str], Any]:
    embeddings = EmbeddingsBuilder(tokenizer, model, normalize=True).from_text(text)
    tokens = build_list_of_tokens_input(text)
    # may end
    result_sources, result_dists = index.get_embeddings_source(embeddings)
    return result_sources, tokens, result_dists
    # expected_count: int = 0
    # dist_sum: float = 0.0
    #
    # links: List[Optional[str]] = []
    #
    # for i, (dist, source) in enumerate(zip(result_dists, result_sources)):
    #
    #     if dist < Config.threshold:
    #         links.append(None)
    #     else:
    #         links.append(source)
    #
    #     if source == expected_url:
    #         expected_count += 1
    #         dist_sum += dist
    #
    # # print(
    # #     f"Got expected URL in {expected_count / len(result_dists) * 100:.4f}% of cases, "
    # #     f"average match distance: {dist_sum / len(result_dists):.4f}"
    # # )
    #
    # dict_with_uniq_colors = build_dict_for_color(links, uniq_color)
    #
    #
    # return build_page_template(text, links, dict_with_uniq_colors)


def build_list_of_tokens_input(text: str) -> list[str]:
    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
    tokens = tokenizer.tokenize(text)

    return tokens

def main(user_input: str) -> tuple[Any, list[str], Any]:
    read_index: bool = True

    if read_index:
        # print("Readings index... ", end='')
        index = Index.load(Config.index_file, Config.mapping_file)
        # print("Done")
    else:
        # print("Index is being built from sources ")
        index = build_index(user_input)

    # print("Test [Data] Searching quotes from the same page:")
    # print('"Childhood w references"')
    return prob_test_wiki_with_colored(index, user_input, childhood_url, 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--userinput", help="User input value", type=str)
    parser.add_argument("--file", help="Quiz file", type=str)
    parser.add_argument("--question", help="Question from quiz", type=str)
    parser.add_argument("--answer", help="Answer for question", type=str)
    parser.add_argument("--usesource", help="Use ready sources", type=str)
    parser.add_argument("--sources", help="Use ready sources", type=str)
    parser.add_argument("--withskip", help="Use 3 max skip in chains", type=str)
    args = parser.parse_args()

    userinput = args.userinput
    file = args.file
    question = args.question
    answer = args.answer
    usesource = args.usesource
    sources = args.sources
    withskip = args.withskip
    result_sources, tokens, result_dists = main(userinput)
    # sentence_length_array, count_colored_token_in_sentence_array, html = main(userinput)

    # dictionary = {
    #     'file': file,
    #     'question': question,
    #     'answer': answer,
    #     'length': sentence_length_array,
    #     'colored': count_colored_token_in_sentence_array,
    #     'html': html
    # }

    float_list = result_dists.tolist()
    str_list = result_sources.tolist()

    dictionary = {
        'file': file,
        'question': question,
        'answer': answer,
        'result_sources': str_list,
        'tokens': tokens,
        'result_dists': float_list,
    }

    json_output = json.dumps(dictionary, indent=4)
    print(json_output)
    # main()
