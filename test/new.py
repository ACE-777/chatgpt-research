import copy
import math
import time
import argparse
import json
from typing import Tuple, List, Set, Optional, Dict

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer, model = Roberta.get_default()
modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
batched_token_ids = torch.empty((1, 512), dtype=torch.int)

page_template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
    <link rel="stylesheet" type="text/css" href="../static/style_result.css">
</head>
<body>
<h1>Result of research</h1>
<pre><b>Input text:</b></pre>
{{ gpt_response }}
<pre><b>Top paragraphs:</b></pre>
{{ list_of_colors }}
<pre><b>Result:</b></pre>
{{ result }}
</body>
</html>
"""

source_link_template_str = "<a href=\"{{ link }}\" class=\"{{ color }}\" title=\"score: {{score}}\">{{ token }}</a>\n"
source_text_template_str = "<a class=\"{{ color }}\"><i>{{ token }}</i></a>\n"
source_item_str = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ link }}</a></br>\n"


class Chain:
    likelihoods: List[float]
    positions: List[int]
    source: str

    def __init__(self, likelihoods: List[float], positions: List[int], source: str):
        assert (len(likelihoods) == len(positions))
        self.likelihoods = likelihoods
        self.positions = positions
        self.source = source

    def __len__(self) -> int:
        return len(self.positions)

    def __str__(self) -> str:
        return (f"Chain {{ pos: {self.positions}, likelihoods: {self.likelihoods}, "
                f"score: {self.get_score()}, source: {self.source} }}")

    def __repr__(self) -> str:
        return str(self)

    def extend(self, likelihood: float, position: int) -> "Chain":
        return Chain(self.likelihoods + [likelihood],
                     self.positions + [position],
                     self.source)

    def get_score(self):
        l = len(self)

        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len) - score
        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / l
        score *= math.log2(2 + l)
        return score


# GLOBAL result sequence:
result_chains: List[Chain] = []


def generate_sequences(chain: Chain, last_hidden_state: torch.Tensor, probs: torch.Tensor,
                       start_idx: int, tokens: List[int], token_pos: int):
    if start_idx >= len(last_hidden_state) or token_pos >= len(tokens):
        if len(chain) > 1:
            result_chains.append(chain)
        return

    for idx in range(start_idx, len(last_hidden_state)):
        token_curr = tokens[token_pos]
        prob = probs[idx][token_curr].item()
        if prob >= 0.05:
            current_chain = chain.extend(prob, token_pos)
            generate_sequences(current_chain, last_hidden_state, probs,
                               idx + 1, tokens, token_pos + 1)
        else:
            if len(chain) > 1:
                result_chains.append(chain)


def main(gpt_response) -> tuple[list[int], list[int]]:
    start_time = time.time()
    index = Index.load(Config.index_file, Config.mapping_file)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения функции1:", execution_time, "секунд")
    start_time = time.time()
    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения функции2:", execution_time, "секунд")
    start_time = time.time()
    faiss.normalize_L2(embeddings)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Время выполнения функции3:", execution_time, "секунд")

    sources, result_dists = index.get_embeddings_source(embeddings)

    gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт

    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = parse_json_to_dict("./artifacts/scrape_wiki.json")

    for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):
        wiki_text = wiki_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()
        print(len(wiki_text))
        print(len(wiki_token_ids))

        for batch in range(0, len(wiki_token_ids), 511):
            wiki_token_ids_batch = wiki_token_ids[batch:batch + 512].unsqueeze(0)
            print("len:::", wiki_token_ids_batch.shape)
            with torch.no_grad():
                print("wiki_token_ids_batch:", wiki_token_ids_batch)
                output_page = modelMLM(wiki_token_ids_batch)
                print("output_page_len:::", len(output_page))
                print("output_page:", output_page)

            last_hidden_state = output_page[0].squeeze()  #
            print("last_hidden_state:", last_hidden_state)
            print("last_hidden_state_len:::", last_hidden_state.shape)
            if last_hidden_state.dim() == 1:
                probs = torch.nn.functional.softmax(last_hidden_state, dim=0)
                probs = probs.unsqueeze(0)
                last_hidden_state = last_hidden_state.unsqueeze(0)
            else:
                probs = torch.nn.functional.softmax(last_hidden_state, dim=1)
            #     в for на 128 сразу делаем на все отрезки, софтмаксим, склеиваем все в одну большую маторицу probs, также получаем len(last_hidden_state) + gpt_tokens, gpt_token_ids, sources и откидываем на golang дальнейшие вычисления
            print("probs:", probs)
            print("probs_len:::", probs.shape)

            empty_chain = Chain([], [], source)
            generate_sequences(empty_chain, last_hidden_state, probs, 0, gpt_token_ids, token_pos)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(result_chains, key=lambda x: x.get_score(), reverse=True):
        marked_in_chain = marked_positions.intersection(chain.positions)
        if len(marked_in_chain) == 0:
            marked_positions |= set(chain.positions)
            filtered_chains.append(chain)

    print("output3")
    # ---------------------------------------------------------------------------------------------------------------
    # prepare tokens for coloring
    tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), gpt_tokens)

    # prepare links for coloring
    pos2chain: Dict[int, Chain] = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.positions:
            pos2chain[pos] = chain

    template_page = Template(page_template_str)
    template_link = Template(source_link_template_str)
    template_text = Template(source_text_template_str)
    template_source_item = Template(source_item_str)

    color: int = 7
    output_page: str = ''
    sentence_length: int = 0
    count_colored_token_in_sentence: int = 0
    sentence_length_array = []
    count_colored_token_in_sentence_array = []
    output_source_list: str = ''
    last_chain: Optional[Chain] = None
    for i, key in enumerate(tokens_for_coloring):
        key: str
        if key == '.':
            sentence_length_array.append(sentence_length)
            count_colored_token_in_sentence_array.append(count_colored_token_in_sentence)
            sentence_length = 0
            count_colored_token_in_sentence = 0

        sentence_length += 1
        if i in pos2chain:
            chain = pos2chain[i]
            source = chain.source
            score = chain.get_score()
            if last_chain == chain:
                count_colored_token_in_sentence += 1
                output_page += template_link.render(link=source,
                                                    score=score,
                                                    color="color" + str(color),
                                                    token=key)
            else:
                count_colored_token_in_sentence += 1
                color += 1
                last_chain = chain
                output_source_list += template_source_item.render(link=source,
                                                                  color="color" + str(color))
                output_page += template_link.render(link=source,
                                                    score=score,
                                                    color="color" + str(color),
                                                    token=key)
        else:
            last_chain = None
            output_page += template_text.render(token=key, color="color0")

    output_source_list += '</br>'
    result_html = template_page.render(result=output_page, gpt_response=gpt_response, list_of_colors=output_source_list)

    with open("./server/templates/template_of_result_page.html", "w", encoding="utf-8") as f:
        f.write(result_html)
    return sentence_length_array, count_colored_token_in_sentence_array


def parse_json_to_dict(json_file_path) -> dict:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    my_dict = {}

    for item in data:
        key, value = list(item.items())[0]
        my_dict[key] = value

    return my_dict


if __name__ == "__main__":
    main(
        'The discussion of the history of a language is typically divided into "external history", describing the ethnic, political, social, technological, and other changes that affected the languages, and "internal history", describing the phonological and grammatical changes undergone by the language')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--userinput", help="User input value", type=str)
    # parser.add_argument("--file", help="Quiz file", type=str)
    # parser.add_argument("--question", help="Question from quiz", type=str)
    # parser.add_argument("--answer", help="Answer for question", type=str)
    # args = parser.parse_args()
    #
    # userinput = args.userinput
    # file = args.file
    # question = args.question
    # answer = args.answer
    # sentence_length_array, count_colored_token_in_sentence_array = main(userinput)
    #
    # dictionary = {
    #     'file': file,
    #     'question': question,
    #     'answer': answer,
    #     'length': sentence_length_array,
    #     'colored': count_colored_token_in_sentence_array
    # }
    #
    # json_output = json.dumps(dictionary, indent=4)
    # print(json_output)
