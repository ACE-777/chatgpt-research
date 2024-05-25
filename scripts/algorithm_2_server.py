import copy
from urllib.parse import quote
import argparse
import json
from typing import List, Set, Any, Union, Optional, Dict
import urllib.parse
from unidecode import unidecode

import work  # type: ignore

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template
from src import Roberta, Config, EmbeddingsBuilder, Index
from transformers import RobertaForMaskedLM

tokenizer, model = Roberta.get_default()

page_template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
 	<meta name="viewport" content="width=device-width,initial-scale=1">
    <meta http-equiv="x-ua-compatible" content="crhome=1">
    <link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
	<link href="https://fonts.googleapis.com/css2?family=Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;1,100;1,300;1,400;1,700;1,900&family=Platypi:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="../static/style_result_updated.css">
</head>
<body>
<div class="topper">
  <a href="/home/" class="back-button">Back</a>
</div>
<div class="container">
	<div class="item">
		<h3>Colored percentage: {{ counter_of_colored }} %</h3>
	</div>
	<div class="item">
		<h3>Result</h3>
		{{ result }}
	</div>
	<div class="item">
		<h3>Top paragraphs</h3>
		{{ list_of_colors }}
	</div>
	<div class="item">
		<h3>Input text</h3>
		{{ gpt_response }}
	</div>
</div>
</body>
</html>
"""

source_link_template_str = "<a href='{{ link }}\' class=\"{{ color }}\" title=\"score: {{score}} ;skip: {{skip}}; " \
                           "iteration_of_skipping: {{count_of_skipping}}\">{{ token }}</a>\n"

source_text_template_str = "<a class=\"{{ color }}\">{{ token }}</a>\n"
source_item_str = "<div class=\"item_paragraphes\">" \
                  "<a href='{{ link }}' class=\"{{ color }}\">{{ link }}</a>" \
                  "</div>\n"


class Chain:
    likelihoods: List[float]
    positions: Optional[List[int]]
    source: str
    source_with_direction_to_text: str
    buffer_source_with_direction_text: str
    skip: int
    count_of_skipping: int
    buffer_positions: List[int]
    buffer_likelihoods: List[float]
    flagSkip: False

    def __init__(self, source: str, likelihoods: Optional[List[float]] = None, positions: Optional[int] = None):
        self.likelihoods = [] if (likelihoods is None) else likelihoods
        self.positions = [] if (likelihoods is None) else likelihoods
        self.source = source
        self.source_with_direction_to_text = quote(source, safe=':/')
        self.buffer_source_with_direction_text = ""
        self.skip = 0
        self.count_of_skipping = 0
        self.buffer_positions = []
        self.buffer_likelihoods = []
        self.flagSkip = False

    def __len__(self) -> int:
        return len(self.positions)

    def __str__(self) -> str:
        return (f"Chain {{ pos: {self.positions}, likelihoods: {self.likelihoods}, "
                f"score: {self.get_score()}, source: {self.source}, skip: {self.skip}, "
                f"iteration_of_skipping: {self.count_of_skipping}, buffer_positions: {self.buffer_positions}, "
                f"buffer_likelihoods: {self.buffer_likelihoods} }}")

    def __repr__(self) -> str:
        return str(self)

    def extend(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        self.positions.append(position)

    def insert_in_begin(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        self.positions.insert(0, position)

    def save_for_probable_insert(self, likelihood: float, position: int) -> None:
        self.buffer_positions.append(position)
        self.buffer_likelihoods.append(likelihood)

    def insert_buffer_in_begin(self):
        for i in range(len(self.buffer_positions)):
            self.insert_in_begin(self.buffer_likelihoods[i], self.buffer_positions[i])

        self.skip = 0
        self.count_of_skipping += 1
        self.buffer_likelihoods = []
        self.buffer_positions = []
        self.flagSkip = False

    def insert_buffer_in_extend(self):
        for i in range(len(self.buffer_positions)):
            self.extend(self.buffer_likelihoods[i], self.buffer_positions[i])

        self.skip = 0
        self.count_of_skipping += 1
        self.buffer_likelihoods = []
        self.buffer_positions = []
        self.flagSkip = False

    def increment_skip(self) -> None:
        self.skip += 1

    def get_score(self):
        l = len(self)

        score = 1.0
        for likelihood in self.likelihoods:
            score *= likelihood

        if l != 0:
            score **= 1 / l

        # first formula
        # score *= math.log2(2 + l)

        # second formula
        # score *= l

        # third formula
        score *= (2 ** l) * (10 ** (self.skip + 3*self.count_of_skipping))

        # fourth formula
        # score = score*(2*l)

        # fourth fifth
        # score = score*(l*(1/2))

        # fourth eight
        # score = score*((1.5)**l)

        return score

    def add_direction_to_text_header(self):
        self.source_with_direction_to_text = self.source_with_direction_to_text + ":~:text="

    def add_direction_to_text_header_begin(self):
        self.source_with_direction_to_text = self.source_with_direction_to_text + self.buffer_source_with_direction_text

    def clear_direction_to_text_header_begin(self, source):
        self.source_with_direction_to_text = source

    def add_direction_of_token_to_text_in_extend(self, wiki_tokens: list[str], token: int):
        if len(wiki_tokens) != token:
            list_of_token = [wiki_tokens[token]]
            self.source_with_direction_to_text = self.source_with_direction_to_text + \
                                                 quote(tokenizer.convert_tokens_to_string(list_of_token))  ## %20

    def add_special_symbol_in_begin_for_direction_to_text_extend(self, wiki_tokens: list[str], token: int):
        if token > 1:
            list_of_token = [wiki_tokens[token - 1]]
            self.source_with_direction_to_text = self.source_with_direction_to_text + \
                                                 tokenizer.convert_tokens_to_string(list_of_token) + "-,"

    def add_special_symbol_in_end_for_direction_to_text_extend(self, wiki_tokens: list[str], token: int):
        if token < len(wiki_tokens) - 1:
            list_of_token = [wiki_tokens[token + 1]]
            if "%20" in quote(tokenizer.convert_tokens_to_string(list_of_token)):
                self.source_with_direction_to_text = self.source_with_direction_to_text + ",-" + \
                                                     quote(tokenizer.convert_tokens_to_string(list_of_token)).split(
                                                         "%20")[1]
            else:
                self.source_with_direction_to_text = self.source_with_direction_to_text + ",-" + \
                                                     quote(tokenizer.convert_tokens_to_string(list_of_token))

    def clear_special_symbol_in_end_for_direction_to_text_extend(self):
        if ",-" in self.source_with_direction_to_text:
            self.source_with_direction_to_text = self.source_with_direction_to_text.split(",-")[0]

    def add_direction_of_token_to_text_in_begin(self, wiki_tokens: list[str], token: int):
        if len(wiki_tokens) != token:
            list_of_token = [wiki_tokens[token]]
            self.buffer_source_with_direction_text = \
                quote(tokenizer.convert_tokens_to_string(list_of_token)) + self.buffer_source_with_direction_text

    def add_special_symbol_in_begin_for_direction_to_text_begin(self, wiki_tokens: list[str], token: int):
        if 1 < token:
            list_of_token = [wiki_tokens[token - 1]]
            self.buffer_source_with_direction_text = tokenizer.convert_tokens_to_string(list_of_token) + "-," + \
                                                     self.buffer_source_with_direction_text

    def add_special_symbol_in_end_for_direction_to_text_begin(self, wiki_tokens: list[str], token: int):
        if token < len(wiki_tokens) - 1:
            list_of_token = [wiki_tokens[token + 1]]
            if "%20" in quote(tokenizer.convert_tokens_to_string(list_of_token)):
                self.buffer_source_with_direction_text = self.buffer_source_with_direction_text + ",-" + \
                                                         quote(tokenizer.convert_tokens_to_string(list_of_token)).split(
                                                             "%20")[1]
            else:
                self.buffer_source_with_direction_text = self.buffer_source_with_direction_text + ",-" + \
                                                         quote(tokenizer.convert_tokens_to_string(list_of_token))

    def clear_special_symbol_in_begin_for_direction_to_text_begin(self):
        if "-," in self.buffer_source_with_direction_text:
            self.buffer_source_with_direction_text = self.buffer_source_with_direction_text.split("-,")[1]


def generate_sequences(source: str, last_hidden_state: int, probs: torch.Tensor, tokens: List[int], token_pos: int,
                       withskip: str, wiki_tokens: List[str]) -> List[Chain]:
    chains_per_token: List[Chain] = []

    for first_idx in range(0, last_hidden_state):
        chain = Chain(source)
        chain.add_direction_to_text_header()
        last_probably_token_in_chain = min(last_hidden_state - first_idx, len(tokens) - token_pos)
        for second_idx in range(0, last_probably_token_in_chain):
            token = first_idx + second_idx
            src_of_token = token_pos + second_idx

            token_curr = tokens[src_of_token]
            prob = probs[token][token_curr].item()
            if prob >= 0.05:
                if chain.flagSkip is True:
                    chain.insert_buffer_in_extend()

                chain.extend(prob, src_of_token)

                if len(chain) > 1:
                    chain.add_direction_of_token_to_text_in_extend(wiki_tokens, token)
                    # chain.add_special_symbol_in_end_for_direction_to_text_extend(wiki_tokens, token)
                    chains_per_token.append(copy.deepcopy(chain))
                    # chain.clear_special_symbol_in_end_for_direction_to_text_extend()

                else:
                    # chain.add_special_symbol_in_begin_for_direction_to_text_extend(wiki_tokens, token)
                    chain.add_direction_of_token_to_text_in_extend(wiki_tokens, token)
            else:
                if withskip == "True" and chain.skip < 3:
                    chain.flagSkip = True
                    chain.increment_skip()
                    chain.save_for_probable_insert(prob, src_of_token)
                else:
                    break

    # then go to left, after went to right. Use be-directional property of MLM
    for last_first_idx in range(0, last_hidden_state):
        chain = Chain(source)
        chain.add_direction_to_text_header()
        last_probably_token_in_chain = min(last_first_idx, token_pos)
        for last_second_idx in range(0, last_probably_token_in_chain):
            token = last_first_idx - last_second_idx
            src_of_token = token_pos - last_second_idx

            token_curr = tokens[src_of_token]
            prob = probs[token][token_curr].item()

            if prob >= 0.05:
                if chain.flagSkip is True:
                    chain.insert_buffer_in_begin()

                chain.insert_in_begin(prob, src_of_token)

                if len(chain) > 1:
                    chain.add_direction_of_token_to_text_in_begin(wiki_tokens, token)
                    # chain.add_special_symbol_in_begin_for_direction_to_text_begin(wiki_tokens, token)
                    chain.add_direction_to_text_header_begin()
                    chains_per_token.append(copy.deepcopy(chain))

                    # chain.clear_special_symbol_in_begin_for_direction_to_text_begin()
                    chain.clear_direction_to_text_header_begin(source)
                    chain.add_direction_to_text_header()
                else:
                    # chain.add_special_symbol_in_end_for_direction_to_text_begin(wiki_tokens, token)
                    chain.add_direction_of_token_to_text_in_begin(wiki_tokens, token)

            else:
                if withskip == "True" and chain.skip < 3:
                    chain.flagSkip = True
                    chain.increment_skip()
                    chain.save_for_probable_insert(prob, src_of_token)

                else:

                    break

    return chains_per_token


def main(gpt_response, withskip) -> str:
    modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = Index.load(Config.index_file, Config.mapping_file)

    embeddings = EmbeddingsBuilder(tokenizer, model, normalize=True).from_text(gpt_response)

    gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт
    sources, result_dists = index.get_embeddings_source(embeddings)
    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = parse_json_to_dict("./artifacts/scrape_wiki.json")
    all_chains_before_sorting = []

    for token_pos, (token, token_id, source, result_dist) in enumerate(zip(gpt_tokens, gpt_token_ids, sources, result_dists)):
        mainSource = copy.deepcopy(source)
        for i in range(0, 5):  # for many source variants per each token
            if result_dist[i] < Config.threshold:
                continue

            source = mainSource[i]
            if source not in wiki_dict:
                continue

            wiki_text = wiki_dict[source]
            wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()
            wiki_tokens = tokenizer.tokenize(wiki_text)
            result_tensor_per_token = torch.empty(0, 50265).to(device)

            for batch in range(0, len(wiki_token_ids), 512):
                wiki_token_ids_batch = wiki_token_ids[batch:batch + 512].unsqueeze(0).to(device)
                with torch.no_grad():
                    modelMLM.to(device)
                    output_page = modelMLM(wiki_token_ids_batch)

                last_hidden_state = output_page[0].squeeze()
                if last_hidden_state.dim() == 1:
                    probs = torch.nn.functional.softmax(last_hidden_state, dim=0).to(device)
                    probs = probs.unsqueeze(0).to(device)
                    last_hidden_state = last_hidden_state.unsqueeze(0)
                else:
                    probs = torch.nn.functional.softmax(last_hidden_state, dim=1).to(device)
                result_tensor_per_token = torch.cat((result_tensor_per_token, probs), dim=0).to(device)
            all_chains_before_sorting += generate_sequences(source, len(result_tensor_per_token),
                                                            result_tensor_per_token, gpt_token_ids, token_pos,
                                                            withskip, wiki_tokens)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(all_chains_before_sorting, key=lambda x: x.get_score(), reverse=True):
        marked_in_chain = marked_positions.intersection(chain.positions)
        if len(marked_in_chain) == 0:
            marked_positions |= set(chain.positions)
            filtered_chains.append(chain)
    tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), gpt_tokens)

    pos2chain: Dict[int, Chain] = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.positions:
            pos2chain[pos] = chain

    template_page = Template(page_template_str)
    template_link = Template(source_link_template_str)
    template_text = Template(source_text_template_str)
    template_source_item = Template(source_item_str)

    color: int = 1
    output_page: str = ''
    sentence_length: int = 0
    count_colored_token_in_sentence: int = 0
    sentence_length_array = []
    count_colored_token_in_sentence_array = []
    output_source_list: str = ''
    last_chain: Optional[Chain] = None
    full_len_of_text = len(gpt_tokens)
    counter_of_colored = 0
    for i, key in enumerate(tokens_for_coloring):
        key: str
        if key.endswith('.') or key == '."':
            sentence_length_array.append(sentence_length)
            count_colored_token_in_sentence_array.append(count_colored_token_in_sentence)
            sentence_length = 0
            count_colored_token_in_sentence = 0

        sentence_length += 1
        if i in pos2chain:
            chain = pos2chain[i]
            source_with_direction_to_text = chain.source_with_direction_to_text
            source = unidecode(urllib.parse.unquote(chain.source.split("https://en.wikipedia.org/wiki/")[1]))
            score = chain.get_score()
            skip = chain.skip
            count_of_skipping = chain.count_of_skipping
            if last_chain == chain:
                counter_of_colored += 1
                count_colored_token_in_sentence += 1
                output_page += template_link.render(link=source_with_direction_to_text,
                                                    score=score,
                                                    skip=skip,
                                                    count_of_skipping=count_of_skipping,
                                                    color="color" + str(color),
                                                    token=key)
            else:
                counter_of_colored += 1
                count_colored_token_in_sentence += 1
                color += 1
                last_chain = chain
                output_source_list += template_source_item.render(link=source,
                                                                  color="color" + str(color))
                output_page += template_link.render(link=source_with_direction_to_text,
                                                    score=score,
                                                    skip=skip,
                                                    count_of_skipping=count_of_skipping,
                                                    color="color" + str(color),
                                                    token=key)
        else:
            last_chain = None
            output_page += template_text.render(token=key, color="color0")

    result_percent_colored = round((counter_of_colored/full_len_of_text)*100)

    result_html = template_page.render(result=output_page, gpt_response=gpt_response, list_of_colors=output_source_list,counter_of_colored=result_percent_colored)

    return result_html


def parse_json_to_dict(json_file_path) -> dict:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    my_dict = {}

    for item in data:
        key, value = list(item.items())[0]
        my_dict[key] = value

    return my_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--userinput", help="User input value", type=str)
    parser.add_argument("--withskip", help="Use 3 max skip in chains", type=str)
    args = parser.parse_args()

    userinput = args.userinput
    withskip = args.withskip

    result_html = main(userinput, withskip)

    dictionary = {
        'html': result_html,
    }

    json_output = json.dumps(dictionary, indent=4)
    print(json_output)
