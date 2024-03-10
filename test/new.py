import copy
import math
import time
import argparse
import json
from test import matrix_pb2
# from matrix_pb2 import MatrixMessages # type: ignore
from typing import Tuple, List, Set, Optional, Dict, Any, Union

import socket
import pickle

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template
from torch import Tensor

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer, model = Roberta.get_default()
modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
batched_token_ids = torch.empty((1, 512), dtype=torch.int)
HOST = '127.0.0.1'
PORT = 9000


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


def generate_sequences(chain: Chain, last_hidden_state: int, probs: torch.Tensor,
                       start_idx: int, tokens: List[int], token_pos: int):
    if start_idx >= last_hidden_state or token_pos >= len(tokens):
        if len(chain) > 1:
            result_chains.append(chain)
        return

    for idx in range(start_idx, last_hidden_state):
        token_curr = tokens[token_pos]
        prob = probs[idx][token_curr].item()
        if prob >= 0.05:
            current_chain = chain.extend(prob, token_pos)
            generate_sequences(current_chain, last_hidden_state, probs,
                               idx + 1, tokens, token_pos + 1)
        else:
            if len(chain) > 1:
                result_chains.append(chain)


def main(gpt_response) -> tuple[list[str], Union[int, list[int]], list[str], list[Any]]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        matrices = [[1.0, 2.0, 3.0]]
        # Отправка массива матриц клиенту
        data = pickle.dumps(matrices)
        conn.sendall(data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        index = Index.load(Config.index_file, Config.mapping_file)

        embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
        faiss.normalize_L2(embeddings)

        sources, result_dists = index.get_embeddings_source(embeddings)

        gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт

        gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

        wiki_dict = parse_json_to_dict("./artifacts/scrape_wiki.json")
        # result_probs_for_each_token = []
        # matrix_msgs = matrix_pb2.MatrixMessages()
        for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):
            wiki_text = wiki_dict[source]
            wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()
            result_tensor_per_token = torch.empty(0, 50265).to(device)
            for batch in range(0, len(wiki_token_ids), 512):
                wiki_token_ids_batch = wiki_token_ids[batch:batch + 512].unsqueeze(0).to(device)
                with torch.no_grad():
                    modelMLM.to(device)
                    output_page = modelMLM(wiki_token_ids_batch)

                last_hidden_state = output_page[0].squeeze()  #
                if last_hidden_state.dim() == 1:
                    probs = torch.nn.functional.softmax(last_hidden_state, dim=0).to(device)
                    probs = probs.unsqueeze(0).to(device)
                    last_hidden_state = last_hidden_state.unsqueeze(0)
                else:
                    probs = torch.nn.functional.softmax(last_hidden_state, dim=1).to(device)
                #     в for на 139 сразу делаем для каждого токена на все отрезки, софтмаксим, склеиваем все в одну большую маторицу probs, также получаем len(probs) + gpt_tokens(только для раскраски), gpt_token_ids, sources и откидываем на golang дальнейшие вычисления
                # в итоге json с массивом из []токен:prob и еще один json с токены: айди_токенов: источники:
                # лучше наверное prob через id в json, чтобы по ним можно было из массивов мета ифнормации ориентирваотся быстрее
                # print("dimaaan:", probs.shape)
                result_tensor_per_token = torch.cat((result_tensor_per_token, probs), dim=0).to(device)
            # result_probs_for_each_token.append(result_tensor_per_token)
            data = pickle.dumps(result_tensor_per_token)
            conn.sendall(data)
        conn.close()

        # print(result_tensor_per_token)
        # empty_chain = Chain([], [], source)
        # last_hidden_state = len(result_tensor_per_token)

        #---
        # matrix_msg = matrix_msgs.matrix_messages.add()
        # for row in result_tensor_per_token:
        #     matrix_msg.row.extend(row.tolist())
        #
        # with open("matrix_messages.bin", "wb") as f:
        #     f.write(matrix_msgs.SerializeToString())

        # generate_sequences(empty_chain, last_hidden_state, result_tensor_per_token, 0, gpt_token_ids, token_pos)

        # result_probs_for_each_token.append(result_tensor_per_token.tolist())

    return gpt_tokens, gpt_token_ids, sources, result_chains
    # filtered_chains: List[Chain] = []
    # marked_positions: Set[int] = set()
    # for chain in sorted(result_chains, key=lambda x: x.get_score(), reverse=True):
    #     marked_in_chain = marked_positions.intersection(chain.positions)
    #     if len(marked_in_chain) == 0:
    #         marked_positions |= set(chain.positions)
    #         filtered_chains.append(chain)
    #
    # print("output3")
    # # ---------------------------------------------------------------------------------------------------------------
    # # prepare tokens for coloring
    # tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), gpt_tokens)
    #
    # # prepare links for coloring
    # pos2chain: Dict[int, Chain] = {}
    # for i, chain in enumerate(filtered_chains):
    #     for pos in chain.positions:
    #         pos2chain[pos] = chain
    #
    # template_page = Template(page_template_str)
    # template_link = Template(source_link_template_str)
    # template_text = Template(source_text_template_str)
    # template_source_item = Template(source_item_str)
    #
    # color: int = 7
    # output_page: str = ''
    # sentence_length: int = 0
    # count_colored_token_in_sentence: int = 0
    # sentence_length_array = []
    # count_colored_token_in_sentence_array = []
    # output_source_list: str = ''
    # last_chain: Optional[Chain] = None
    # for i, key in enumerate(tokens_for_coloring):
    #     key: str
    #     if key == '.':
    #         sentence_length_array.append(sentence_length)
    #         count_colored_token_in_sentence_array.append(count_colored_token_in_sentence)
    #         sentence_length = 0
    #         count_colored_token_in_sentence = 0
    #
    #     sentence_length += 1
    #     if i in pos2chain:
    #         chain = pos2chain[i]
    #         source = chain.source
    #         score = chain.get_score()
    #         if last_chain == chain:
    #             count_colored_token_in_sentence += 1
    #             output_page += template_link.render(link=source,
    #                                                 score=score,
    #                                                 color="color" + str(color),
    #                                                 token=key)
    #         else:
    #             count_colored_token_in_sentence += 1
    #             color += 1
    #             last_chain = chain
    #             output_source_list += template_source_item.render(link=source,
    #                                                               color="color" + str(color))
    #             output_page += template_link.render(link=source,
    #                                                 score=score,
    #                                                 color="color" + str(color),
    #                                                 token=key)
    #     else:
    #         last_chain = None
    #         output_page += template_text.render(token=key, color="color0")
    #
    # output_source_list += '</br>'
    # result_html = template_page.render(result=output_page, gpt_response=gpt_response, list_of_colors=output_source_list)
    #
    # with open("./server/templates/template_of_result_page.html", "w", encoding="utf-8") as f:
    #     f.write(result_html)
    # return sentence_length_array, count_colored_token_in_sentence_array


def parse_json_to_dict(json_file_path) -> dict:
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    my_dict = {}

    for item in data:
        key, value = list(item.items())[0]
        my_dict[key] = value

    return my_dict


if __name__ == "__main__":
    # gpt_tokens, gpt_token_ids, sources, result_chaines_2 = main("National parks were first designated under the National Parks and Access to the Countryside Act 1949, and in England and Wales any new national park is designated under this Act, and must be confirmed by the Secretary of State for Environment, Food and Rural Affairs. The 1949 Act came about after a prolonged campaign for public access to the countryside in the United Kingdom with its roots in the Industrial Revolution. The first 'freedom to roam' bill was introduced to Parliament in 1884 by James Bryce but it was not until 1931 that a government inquiry recommended the creation of a 'National Park Authority' to select areas for designation as national parks. Despite the recommendation and continued lobbying and demonstrations of public discontent, such as the 1932 Kinder Scout mass trespass in the Peak District, nothing further was done until a 1945 white paper on national parks was produced as part of the Labour Party's planned post-war reconstruction, leading in 1949 to the passing of the National Parks and Access to the Countryside Act."+
    # "In England and Wales, as in Scotland, designation as a national park means that the area has been identified as being of importance to the national heritage and as such is worthy of special protection and attention. Unlike the model adopted in many other countries, such as the US and Germany, this does not mean the area is owned by the state. National parks in the United Kingdom may include substantial settlements and human land uses which are often integral parts of the landscape, and within a national park there are many landowners including public bodies and private individuals")
    # print("res:::", result_chaines_2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--userinput", help="User input value", type=str)
    parser.add_argument("--file", help="Quiz file", type=str)
    parser.add_argument("--question", help="Question from quiz", type=str)
    parser.add_argument("--answer", help="Answer for question", type=str)
    args = parser.parse_args()

    userinput = args.userinput
    file = args.file
    question = args.question
    answer = args.answer
    gpt_tokens, gpt_token_ids, sources, result_chains_2 = main(userinput)

    # print(gpt_token_ids)
    # print(len(gpt_token_ids))
    # matrices_dict = {i + 1: matrix for i, matrix in enumerate(result_probs_for_each_token)}
    # print(result_chaines_2)
    dictionary = {
        'file': file,
        'question': question,
        'answer': answer,
        # 'result_probs_for_each_token': matrices_dict,
        'tokens': gpt_tokens,
        'tokens_ids': gpt_token_ids,
        'result_sources': sources
    }

    json_output = json.dumps(dictionary, indent=4)
    print(json_output)
