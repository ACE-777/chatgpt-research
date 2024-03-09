import copy
import math
import time
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
    begin_pos: Optional[int]
    end_pos: Optional[int]
    likelihoods: List[float]
    skips: int = 0
    source: str

    def __init__(self, source: str,
                 begin_pos: Optional[int] = None,
                 end_pod: Optional[int] = None,
                 likelihoods: Optional[List[float]] = None,
                 skips: int = 0):
        self.begin_pos = begin_pos
        self.end_pos = end_pod
        self.likelihoods = [] if (likelihoods is None) else likelihoods
        self.source = source
        self.skips = skips

    def __len__(self) -> int:
        if self.begin_pos is None:
            return 0

        return self.end_pos - self.begin_pos + 1

    def __str__(self) -> str:
        return (f"Chain {{\n"
                f"\tseq = {self.begin_pos}..{self.end_pos}\n"
                f"\tlikelihoods = {self.likelihoods}\n"
                f"\tskips = {self.skips}\n"
                f"\tscore = {self.get_score()}\n"
                f"\tsource = {self.source}\n"
                f"}}\n")

    def __repr__(self) -> str:
        return (f"Chain("
                f"begin_pos={self.begin_pos}, "
                f"end_pos={self.end_pos}, "
                f"likelihoods={self.likelihoods!r}, "
                f"source={self.source!r}, "
                f"skips={self.skips}"
                f")")

    def append(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        if self.begin_pos is None:
            self.begin_pos = position
            self.end_pos = position
        else:
            if self.end_pos + self.skips + 1 != position:
                raise ValueError(f"{self.end_pos=}, {position=}")
            self.end_pos += self.skips + 1
        self.skips = 0

    def skip(self) -> None:
        self.skips += 1

    def get_token_positions(self) -> Set[int]:
        return set(range(self.begin_pos, self.end_pos + 1))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / len(self.likelihoods)
        score *= math.log2(2 + len(self))
        return score


def generate_sequences(source_len: int, likelihoods: torch.Tensor,
                       token_ids: List[int], token_start_pos: int, source: str) -> List[Chain]:
    result_chains: List[Chain] = []

    for source_start_pos in range(0, source_len):
        chain = Chain(source)
        shift_upper_bound = min(source_len - source_start_pos, len(token_ids) - token_start_pos)
        for shift in range(0, shift_upper_bound):
            token_pos = token_start_pos + shift
            source_pos = source_start_pos + shift

            assert token_pos < len(token_ids)
            assert source_pos < source_len

            token_curr_id = token_ids[token_pos]
            token_curr_likelihood = likelihoods[source_pos][token_curr_id].item()

            if token_curr_likelihood < 1e-5:
                chain.skip()
                if chain.skips > 3:
                    break
            else:
                chain.append(token_curr_likelihood, token_pos)
                if len(chain) > 1:
                    result_chains.append(copy.deepcopy(chain))

    return result_chains


def main(gpt_response) -> None:
    index = Index.load(Config.index_file, Config.mapping_file)

    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    print(embeddings)
    faiss.normalize_L2(embeddings)

    sources, result_dists = index.get_embeddings_source(embeddings)
    print("soureces:", sources, "result_ids dists:", result_dists, "\n\n")

    gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт
    print("tokens:", gpt_tokens, "\n\n")  # все токены разбитые из input

    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = dict()
    for page in Config.page_names:
        wiki_dict |= Wiki.parse(page)

    start = time.perf_counter()
    result_chains = []
    for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):
        wiki_text = wiki_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()

        print(f"> token: '{token}', id: {token_id}, source token count: {len(wiki_token_ids)}, top source: {source}, ")
        for batch in range(0, len(wiki_token_ids), 256):
            print(f"\tbatch: [{batch} : {batch + 512})")
            wiki_token_ids_batch = wiki_token_ids[batch:batch + 512]
            if len(wiki_token_ids_batch) < 2:
                break

            wiki_token_ids_batch = wiki_token_ids_batch.unsqueeze(0)

            with torch.no_grad():
                output_page = modelMLM(wiki_token_ids_batch)

            wiki_logits = output_page[0].squeeze()
            likelihoods = torch.nn.functional.softmax(wiki_logits, dim=1)
            result_chains += generate_sequences(len(wiki_logits), likelihoods, gpt_token_ids, token_pos, source)

    print("All sequences: ")
    for chain in result_chains:
        print(chain)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(result_chains, key=lambda x: x.get_score(), reverse=True):
        positions = chain.get_token_positions()
        marked_positions_inside_chain = marked_positions.intersection(positions)
        if len(marked_positions_inside_chain) == 0:
            marked_positions |= positions
            filtered_chains.append(chain)

    print("Filtered chains:")
    for chain in filtered_chains:
        print(chain)

    print(f"Time: {time.perf_counter() - start} s.")

    # prepare tokens for coloring
    tokens_for_coloring = map(lambda s: tokenizer.convert_tokens_to_string([s]), gpt_tokens)

    # prepare links for coloring
    pos2chain: Dict[int, Chain] = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.get_token_positions():
            pos2chain[pos] = chain

    template_page = Template(page_template_str)
    template_link = Template(source_link_template_str)
    template_text = Template(source_text_template_str)
    template_source_item = Template(source_item_str)

    color: int = 7
    output_page: str = ''
    output_source_list: str = ''
    last_chain: Optional[Chain] = None
    for i, key in enumerate(tokens_for_coloring):
        key: str

        if i in pos2chain:
            chain = pos2chain[i]
            source = chain.source
            score = chain.get_score()
            if last_chain == chain:
                output_page += template_link.render(link=source,
                                                    score=score,
                                                    color="color" + str(color),
                                                    token=key)
            else:
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


if __name__ == "__main__":

    main(
        "Presley's father Vernon was of German, Scottish, and English origins, and a descendant of the Harrison family "
        "of Virginia through his mother, Minnie Mae Presley (née Hood). Presley's mother Gladys was Scots-Irish with "
        "some French Norman ancestry. She and the rest of the family believed that her great-great-grandmother,"
        " Morning Dove White, was Cherokee. This belief was restated by Elvis's granddaughter Riley Keough in 2017. "
        "Elaine Dundy, in her biography, supports the belief.")  # gpt output
