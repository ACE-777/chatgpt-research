from typing import Union, Tuple, List, Any

import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

from .SourceMapping import SourceMapping
from .Roberta import Roberta
from .EmbeddingsBuilder import EmbeddingsBuilder
from .Config import Config

__all__ = ['Index']


class Index:
    def __init__(self, index: faiss.Index, mapping: SourceMapping, threshold: float = Config.threshold):
        self.index: faiss.Index = index
        self.mapping: SourceMapping = mapping
        self.threshold = threshold

    @staticmethod
    def load(index_file: str,
             mapping_file: str,
             threshold: float = Config.threshold,
             use_gpu: bool = Config.faiss_use_gpu) -> "Index":

        faiss_index = faiss.read_index(index_file)
        if use_gpu:
            faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)

        mapping = SourceMapping.read_csv(mapping_file)
        return Index(faiss_index, mapping, threshold)

    def save(self, index_file: str, mapping_file: str) -> None:
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), index_file)
        self.mapping.to_csv(mapping_file)

    @staticmethod
    def from_embeddings(embeddings: Union[np.ndarray, pd.DataFrame],
                        mapping: SourceMapping,
                        threshold: float = Config.threshold,
                        use_gpu: bool = Config.faiss_use_gpu) -> "Index":
        """
        Builds index from provided embeddings
        :param embeddings: data to build the index
        :param threshold: threshold to divide data
        :param mapping: index to source mapping
        :param use_gpu: if set, GPU is used to build the index
        :return: IndexFlatIP, or GpuIndexFlatIP id use_gpu is True
        """
        # C-contiguous order and np.float32 type are required
        if isinstance(embeddings, np.ndarray) and embeddings.flags['C_CONTIGUOUS']:
            data = embeddings.astype(np.float32)
        else:
            data = np.array(embeddings, order="C", dtype=np.float32)

        sequence_len, embedding_len = data.shape

        faiss.normalize_L2(data)
        print("Building index... ", end="")
        index = faiss.IndexFlatIP(embedding_len)
        if use_gpu:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        index.add(data)
        print("Done")

        return Index(index, mapping, threshold)
    @staticmethod
    def from_embeddings_input(embeddings: Union[np.ndarray, pd.DataFrame],
                        mapping: SourceMapping,
                        threshold: float = Config.threshold,
                        use_gpu: bool = Config.faiss_use_gpu) -> "Index":
        """
        Builds index from provided embeddings
        :param embeddings: data to build the index
        :param threshold: threshold to divide data
        :param mapping: index to source mapping
        :param use_gpu: if set, GPU is used to build the index
        :return: IndexFlatIP, or GpuIndexFlatIP id use_gpu is True
        """
        # C-contiguous order and np.float32 type are required
        if isinstance(embeddings, np.ndarray) and embeddings.flags['C_CONTIGUOUS']:
            data = embeddings.astype(np.float32)
        else:
            data = np.array(embeddings, order="C", dtype=np.float32)

        sequence_len, embedding_len = data.shape

        faiss.normalize_L2(data)
        index = faiss.IndexFlatIP(embedding_len)
        if use_gpu:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        index.add(data)

        return Index(index, mapping, threshold)

    @staticmethod
    def from_config_wiki(normalization: bool,centroid_file: Any):
        """
        generates index from online Wikipedia
        Refer to Embeddings.from_wiki()
        :return:
        """
        embeddings = EmbeddingsBuilder(*Roberta.get_default(), normalize=normalization, centroid_file=centroid_file)
        return Index.from_embeddings(*embeddings.from_wiki())

    @staticmethod
    def from_config_wiki_input(normalization: bool,centroid_file: Any, articles: list[str]):
        """
        generates index from online Wikipedia
        Refer to Embeddings.from_wiki()
        :return:
        """
        embeddings = EmbeddingsBuilder(*Roberta.get_default(), normalize=normalization, centroid_file=centroid_file)
        return Index.from_embeddings_input(*embeddings.from_wiki_input(articles))

    def dim(self):
        return self.index.d

    def search(self, x: np.ndarray) :
        """
        :param x: 2D array with shape (N, dim)
        :return: tuple(indexes, distances)
        """
        # dists, ids = self.index.search(x, 1)
        # print("first:",  dists, ids)
        dists, ids = self.index.search(x, 10)
        # print("second:", dists, ids)
        # for i, (dist, ids) in enumerate(zip(dists, ids)):
        ids = np.squeeze(ids)
        dists = np.squeeze(dists)

        return ids, dists


    def get_source(self, idx: int):
        return self.mapping.get_source(idx)

    def get_embeddings_source(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: 2D array with shape (N, dim)
        :return: tuple(source_strings, distances)
        """
        indexes, distances = self.search(x)
        indexes_with_source = np.empty(indexes.shape, dtype=object)
        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                source = self.get_source(indexes[i][j])
                indexes_with_source[i][j] = source

        # result = [[self.get_source(element) for element in sublist] for sublist in indexes]
        # return list(map(lambda i: list(map(self.get_source, i)), indexes)), distances
        return indexes_with_source, distances
