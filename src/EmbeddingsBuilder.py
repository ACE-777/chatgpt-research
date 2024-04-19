import faiss  # type: ignore
from progress.bar import ChargingBar  # type: ignore
import torch  # type: ignore
import numpy as np
import time

from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from typing import List, Tuple, Optional, Callable, Dict

from .SourceMapping import SourceMapping
from .Config import Config
from .Wiki import Wiki


class EmbeddingsBuilder:
    def __init__(self,
                 tokenizer: RobertaTokenizer,
                 model: RobertaModel,
                 normalize: bool = False,
                 centroid_file: Optional[str] = Config.centroid_file):
        self.tokenizer = tokenizer
        self.model = model
        self.max_sequence_length = self.model.config.max_position_embeddings
        self.embedding_length = self.model.config.hidden_size
        self.normalize = normalize
        # print(f"Embedding normalization: {normalize}")

        self.suppress_progress = False

        if centroid_file is not None:
            self.centroid = np.load(Config.centroid_file)
            # print('Centroid loaded!')
        else:
            self.centroid = np.zeros(self.embedding_length)
            # print('No centroid')

    def from_ids(self, input_ids: List[int]) -> np.ndarray:
        """
        :param input_ids: list of token ids
        :return: numpy array with dimensions (token_count, embedding_length)
        """
        sequence_length: int = self.max_sequence_length
        shorten_by: int = 2 if sequence_length % 2 == 0 else 3
        sequence_length -= shorten_by
        window_step: int = sequence_length // 2

        embedding_len: int = self.embedding_length
        embeddings: np.ndarray = np.empty((0, embedding_len))
        previous_half: Optional[np.ndarray] = None
        # previous_half: np.ndarray | None = None

        iterable = range(0, len(input_ids), window_step)
        if not self.suppress_progress:
            iterable = ChargingBar('Embeddings').iter(iterable)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # start_time = time.time()
        """
        going with sliding window throw whole text.
        for example if sequence_length is 512 and window step is 256 we will start from [0;512]->[256;768]->[512;1024]..
        """
        for i in iterable:
            # Create tensor with acceptable dimensions:
            input_ids_tensor = torch.tensor(input_ids[i: i + sequence_length]).unsqueeze(0).to(device)

            # # Moves tensor to model's device
            # input_ids_tensor = input_ids_tensor.to(self.model.device)

            output = self.model(input_ids_tensor)
            seq_embeddings = output.last_hidden_state.detach().squeeze(0).cpu().numpy().astype(np.float32)

            if previous_half is not None:
                # Get mean value of 2 halves (prev[:t] and curr[t:])
                current_half = (previous_half + seq_embeddings[:window_step]) / 2
                embeddings = np.concatenate([embeddings, current_half])
            else:
                embeddings = seq_embeddings[:window_step]

            previous_half = seq_embeddings[window_step:]
        # end_time = time.time()
        # execution_time = end_time - start_time

        # print("Время выполнения функции:", execution_time, "секунд")
        if previous_half is not None:
            embeddings = np.concatenate([embeddings, previous_half])

        count, length = embeddings.shape
        assert count == len(input_ids)
        assert length == embedding_len
        # print("centroid;;::", self.centroid)
        # print("centroidshape;;::", self.centroid.shape)
        # print("embedings11:", embeddings)
        # print("embedddshape;;::", embeddings.shape)

        # print("centroida:",  self.centroid)
        embeddings -= self.centroid
        if self.normalize:
            faiss.normalize_L2(embeddings)


        # print("embedings22:", embeddings)
        return embeddings

    def from_text(self, text: str) -> np.ndarray:
        """
        :param text: input
        :return: numpy array with dimensions (token_count, embedding_length)
                 token_count depends on text contents
        """

        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return self.from_ids(input_ids)

    def from_wiki(self) -> Tuple[np.ndarray, SourceMapping]:
        """
        Computes embeddings for online Wikipedia
        Target pages are specified via config variable `page_names`
        :return: Tuple:
                    - Embeddings as 2d numpy.array
                    - and Interval to Source mapping
        """
        source_list = Config.page_names
        return self.from_sources(source_list, Wiki.parse)

    def from_sources(self, source_list: List[str], source_provider: Callable[[str], Dict[str, str]])\
            -> Tuple[np.ndarray, SourceMapping]:
        """
        Computes embeddings for online Wikipedia
        Target pages are specified via config variable `page_names`
        :param source_list: Name of the source to be passed to source_provider
        :param source_provider: Function that accepts source name and
                                returns dictionary sub_source -> source_text
        :return: Tuple:
                    - Embeddings as 2d numpy.array
                    - and Interval to Source mapping
        """
        src_map = SourceMapping()
        embeddings = np.empty((0, self.model.config.hidden_size))

        for i, page in enumerate(source_list):
            print(f"Source {i + 1}/{len(source_list)} in processing")
            sources_dict = source_provider(page)

            input_ids: List[int] = []
            for title, text in sources_dict.items():
                tokens = self.tokenizer.tokenize(text)
                input_ids += self.tokenizer.convert_tokens_to_ids(tokens)
                src_map.append_interval(len(tokens), title)

            page_embeddings = self.from_ids(input_ids)
            embeddings = np.concatenate([embeddings, page_embeddings])
        print("embeddings:", embeddings)
        return embeddings, src_map
