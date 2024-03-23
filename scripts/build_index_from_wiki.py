from src import Index, Config


def main() -> None:
    index = Index.from_config_wiki(True, Config.centroid_file)
    index.save(Config.index_file, Config.mapping_file)


if __name__ == '__main__':
    main()
