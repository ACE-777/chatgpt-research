"""
Arg: Text
Description
 * Find close sources in Source index (
"""
import sys
from typing import Dict

# import lucene
from search import Searcher
from src import Config, Roberta, EmbeddingsBuilder, Index


def build_index(text: str) -> Index:
    # lucene.getVMEnv().attachCurrentThread()
    with Searcher(Config.source_index_path, 100) as searcher:
        tokens = searcher.split_text(text)

        print("Creating query")
        for window_size in range(1, 6):
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i: i + window_size]
                searcher.add_clause(window)

        print("Searching...")
        sources = searcher.search()

    titles = list(map(lambda source: source.title, sources))
    bodies = map(lambda source: source.body, sources)
    source_dict = dict(zip(titles, bodies))

    def source_provider(title: str) -> Dict[str, str]:
        wikilink = f"https://en.wikipedia.org/wiki/{title}"
        return {wikilink: source_dict[title]}

    builder = EmbeddingsBuilder(*Roberta.get_default(), normalize=True)
    embeddings, mapping = builder.from_sources(titles, source_provider)
    return Index.from_embeddings(embeddings, mapping)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Supply text to analyze as first argument")
    index = build_index(sys.argv[1])
    index.save(Config.temp_index_file, Config.temp_mapping_file)
