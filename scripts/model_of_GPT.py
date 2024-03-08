import openai
import api_key  # type: ignore

from jinja2 import Template
from transformers import RobertaTokenizer  # type: ignore
from typing import Dict, Iterable, Tuple

from src import Config

openai.api_key = api_key.key
max_tokens = 128
model_engine = "text-davinci-003"

page_template = """
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

link_template = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ token }}</a>"
list_of_articles = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ token }}</a></br>"


def model(message_from_user: str):
    """
    Creates a model based on Chat-GPT for the answering users questions.
    """
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=message_from_user,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    print(completion.choices[0].text)
    # build_page_template(completion.choices[0].text)


def list_of_colors(dict_with_uniq_colors: Dict[str, str]) -> str:
    template = Template(list_of_articles)
    output = ''
    for key, value in dict_with_uniq_colors.items():
        output += template.render(link=key, color=value, token=key)

    output += '</br>'
    return output


def build_list_of_tokens_input(text: str) -> list[str]:
    tokenizer = RobertaTokenizer.from_pretrained(Config.model_name)
    tokens = tokenizer.tokenize(text)

    return tokens


def build_link_template(tokens: Iterable[str], source_link: list[str], dict_with_uniq_colors: Dict[str, str]) -> tuple[str, list[int],  list[int]]:
    template = Template(link_template)
    tokens = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), tokens)
    output = ''
    sentence_length = 0
    count_colored_token_in_sentence = 0
    sentence_length_array = []
    count_colored_token_in_sentence_array = []
    for i, (key, src) in enumerate(zip(tokens, source_link)):
        flag = False
        if key == '.':
            sentence_length_array.append(sentence_length)
            count_colored_token_in_sentence_array.append(count_colored_token_in_sentence)
            sentence_length = 0
            count_colored_token_in_sentence = 0

        sentence_length += 1
        if src is not None:
            for link_color, color in dict_with_uniq_colors.items():
                if src == link_color:
                    output += template.render(link=src, color=color, token=key)
                    flag = True
                    count_colored_token_in_sentence+=1
                    continue
            if not flag:
                if i % 2 != 0:
                    count_colored_token_in_sentence += 1
                    output += template.render(link=src, color="color7", token=key)
                else:
                    count_colored_token_in_sentence += 1
                    output += template.render(link=src, color="color8", token=key)
        else:
            output += template.render(token=key, color="color0")

    return output, sentence_length_array, count_colored_token_in_sentence_array


def build_page_template(completion: str, source_links: list[str], dict_with_uniq_colors: Dict[str, str]) -> \
        tuple[ list[int],  list[int], str]:
    template = Template(page_template)
    tokens_from_output = build_list_of_tokens_input(completion)  # can integrate chatgpt response
    result_of_color, sentence_length_array, count_colored_token_in_sentence_array = \
        build_link_template(tokens_from_output, source_links, dict_with_uniq_colors)
    result_of_list_of_colors = list_of_colors(dict_with_uniq_colors)
    result_html = template.render(result=result_of_color, gpt_response=completion,
                                  list_of_colors=result_of_list_of_colors)
    # with open("./server/templates/template_of_result_page.html", "w", encoding="utf-8") as f:
    #     f.write(result_html)
    return sentence_length_array, count_colored_token_in_sentence_array, result_html


if __name__ == "__main__":
    model("when was born Elvis?")
