import wikipediaapi  # type: ignore
from src import Config, Wiki
import json

if __name__ == "__main__":
    wiki_dict = dict()
    for page in Config.page_names:
        wiki_dict |= Wiki.parse(page)
    #     --------------------------------------------------------
    keys = list(wiki_dict.keys())
    values = [wiki_dict[key] for key in keys]
    key_value_pairs = [{key: value} for key, value in zip(keys, values)]
    with open("./artifacts/scrape_wiki.json", "w") as json_file:
        json.dump(key_value_pairs, json_file, indent=4)
    # print(len(wiki_dict))
    # iterator = iter(wiki_dict)
    #
    # # Получаем первый ключ
    # first_key = next(iterator)
    #
    # print("Первый ключ словаря:", first_key)
    #
    # first_value = wiki_dict[first_key]
    # print("Значение, соответствующее первому ключу:", first_value)
