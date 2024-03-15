import copy
import math
import time
import argparse
import json
from typing import List, Set, Any, Union, Tuple, Optional, Dict

import work # type: ignore

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template
from src import Roberta, Config, Embeddings, Index
from transformers import  RobertaForMaskedLM


tokenizer, model = Roberta.get_default()
modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
batched_token_ids = torch.empty((1, 512), dtype=torch.int)


page_template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
    <link rel="stylesheet" type="text/css" href="../cmd/static/style_result.css">
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
    positions: Optional[List[int]]
    source: str

    def __init__(self,source: str, likelihoods: Optional[List[float]] = None, positions: Optional[int] = None):
        self.likelihoods = [] if (likelihoods is None) else likelihoods
        self.positions = [] if (likelihoods is None) else likelihoods
        self.source = source

    def __len__(self) -> int:
        return len(self.positions)

    def __str__(self) -> str:
        return (f"Chain {{ pos: {self.positions}, likelihoods: {self.likelihoods}, "
                f"score: {self.get_score()}, source: {self.source} }}")

    def __repr__(self) -> str:
        return str(self)

    def extend(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        self.positions.append(position)
        # return Chain(self.likelihoods + [likelihood],
        #              self.positions + [position],
        #              self.source)

    def get_score(self):
        l = len(self)

        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / l
        score *= math.log2(2 + l)
        return score



def generate_sequences(source: str, last_hidden_state: int, probs: torch.Tensor, tokens: List[int], token_pos: int)-> List[Chain]:
    chains_per_token: List[Chain] = []

    for first_idx in range(0, last_hidden_state):
        chain = Chain(source)
        last_probably_token_in_chain = min(last_hidden_state - first_idx, len(tokens) - token_pos)

        for second_idx in range(0, last_probably_token_in_chain):
            token = first_idx + second_idx
            src_of_token = token_pos + second_idx

            token_curr = tokens[src_of_token]
            prob = probs[token][token_curr].item()

            if prob >= 0.05:
                chain.extend(prob, src_of_token)
                if len(chain) > 1:
                    chains_per_token.append(copy.deepcopy(chain))
            else:
                print("prob:",prob)
                print("len(tokens)", len(tokens))
                print("token_pos:", token_pos)
                print("last_hidden_state:", last_hidden_state)
                print("first_idx", first_idx)
                print("last_probably_token_in_chain:", last_probably_token_in_chain)
                break

    return chains_per_token


def main(gpt_response, use_source, sources_from_input) -> tuple[list[str], list[str], list[Chain], Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = Index.load(Config.index_file, Config.mapping_file)

    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    faiss.normalize_L2(embeddings)

    gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт

    if use_source == "True":
        sources_from_input = [link.strip() for link in sources_from_input.split(',')]

        iteration_of_sentence=0
        sources=[]
        for token in range(0, len(gpt_tokens)):
            if token == len(gpt_tokens):
                break

            if gpt_tokens[token] == "'.'":
                iteration_of_sentence+=1

            sources.append(sources_from_input[iteration_of_sentence])
    else:
        sources, result_dists = index.get_embeddings_source(embeddings)

    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = parse_json_to_dict("./artifacts/scrape_wiki.json")
    all_chains_before_sorting=[]
    # start_time = time.time()
    for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):
        wiki_text = wiki_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()
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
        all_chains_before_sorting += generate_sequences(source, len(result_tensor_per_token), result_tensor_per_token, gpt_token_ids, token_pos)

    # end_time = time.time()
    print("==================")
    print("unfiltered chains:", all_chains_before_sorting)
    print("==================")
    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(all_chains_before_sorting, key=lambda x: x.get_score(), reverse=True):
        marked_in_chain = marked_positions.intersection(chain.positions)
        if len(marked_in_chain) == 0:
            marked_positions |= set(chain.positions)
            filtered_chains.append(chain)
    print("filtered_chains::::", filtered_chains)
    tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), gpt_tokens)
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

    return gpt_tokens, sources, filtered_chains, result_html



def parse_json_to_dict(json_file_path) -> dict:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    my_dict = {}

    for item in data:
        key, value = list(item.items())[0]
        my_dict[key] = value

    return my_dict


if __name__ == "__main__":
    gpt_tokens, sources, result_chaines_2, result_html = main(
        "In England and Wales, any new national park is named under the National Parks and Access to the Countryside Act 1949, and its designation is subject to confirmation by the Secretary of State for Environment, Food, and Rural Affairs. National parks were initially established under this Act. The Industrial Revolution in the United Kingdom gave rise to a protracted struggle for public access to the countryside, which culminated in the 1949 Act. James Bryce submitted the first 'freedom to roam' bill in Parliament in 1884, but it wasn't until 1931 that a government investigation suggested establishing a 'National Park Authority' to choose which regions to declare national parks.", "True", "https://en.wikipedia.org/wiki/National_parks_of_the_United_Kingdom#Legal_designation,https://en.wikipedia.org/wiki/National_parks_of_the_United_Kingdom#Legal_designation,https://en.wikipedia.org/wiki/National_parks_of_the_United_Kingdom#Legal_designation,https://en.wikipedia.org/wiki/National_parks_of_the_United_Kingdom#Legal_designation")
    file="ggg"
    question="qqqq"
    answer="aaaa"
    use_source = False

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--userinput", help="User input value", type=str)
    # parser.add_argument("--file", help="Quiz file", type=str)
    # parser.add_argument("--question", help="Question from quiz", type=str)
    # parser.add_argument("--answer", help="Answer for question", type=str)
    # parser.add_argument("--usesource", help="Use ready sources", type=str)
    # parser.add_argument("--sources", help="Use ready sources", type=str)
    # args = parser.parse_args()
    #
    # userinput = args.userinput
    # file = args.file
    # question = args.question
    # answer = args.answer
    # usesource = args.usesource
    # sources = args.sources
    # gpt_tokens, sources_res, result_chaines_2, result_html = main(userinput, usesource, sources)
    #
    # json_data_chains = json.dumps([chain.__dict__ for chain in result_chaines_2])
    #
    # dictionary = {
    #     'file': file,
    #     'question': question,
    #     'answer': answer,
    #     'tokens': gpt_tokens,
    #     'result_sources': sources_res,
    #     'chains': json_data_chains,
    #     'html': result_html,
    # }
    #
    # json_output = json.dumps(dictionary, indent=4)
    # print(json_output)