from abc import ABC, abstractmethod
import pickle
from typing import BinaryIO, Iterator
from pathlib import Path
import os

class Tokenizer(ABC):
    """
    Base class of Tokenizers
    """
    def __init__(self, input_path: str | None, vocab_size: int | None, special_tokens: list[str] | None):
        self.input_path = input_path

        # self.eof_token = None
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = []
        self.PAT = self._default_pattern()

        # init vocab
        self._init_vocab()
        # init vocab sizes
        self.cur_vsize = len(self.vocab)
        self.vocab_size = vocab_size

        # init merges
        self.merges = {}

    def _init_vocab(self):
        # init vocab based on special tokens
        self.vocab = {i: st.encode("utf-8") for i, st in enumerate(self.special_tokens)}
        self.tokens2id = {st.encode("utf-8"):i  for i, st in enumerate(self.special_tokens)}
        self.cur_vsize = len(self.vocab)
        # get id for end of file token (eof)
        self.eof_token = None
        for i, k in self.vocab.items():
            if k == "<|endoftext|>".encode("utf-8"):
                self.eof_token = i
                break
        # add to vocab main 256 bytes
        for i in range(256):
            self.vocab[self.cur_vsize + i] = bytes([i])
            self.tokens2id[bytes([i])] = self.cur_vsize + i

    def _default_pattern(self):
        return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def find_chunk_boundaries(
        self,
        file: BinaryIO, 
        desired_num_chunks: int
    ) -> list[int]:
        """
        NOTE: part of Stanford CS336
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.

        Returns:
            list[int]: File byte offsets for chunk boundaries.
        """
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks 

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = [mini_chunk.find(st.encode("utf-8")) for st in self.special_tokens if mini_chunk.find(st.encode("utf-8")) != -1]
                if found_at != []:
                    chunk_boundaries[bi] = initial_position + min(found_at)
                    break
                initial_position += mini_chunk_size
        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def from_files(self, vocab_path: str, merges_path: str | None = None):
        """
        Construct the following from a serialized objects (pickle):
            - vocab: dict[int, bytes]
            - tokens2id: dict[bytes, int]
            - merges: list[(bytes, bytes)]

        Args:
            vocab_path (str): A path to the Vocabulary file.
            merges_path (str): A path to the Merges file.
        """
        # deserialize vocab
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        # create tokens2id
        self.tokens2id = dict()
        for i, t in self.vocab.items():
            self.tokens2id[t] = i
        # update vocab size
        self.vocab_size = len(self.vocab)
        # deserialize merges
        if merges_path:
            with open(merges_path, "rb") as f:
                merges = pickle.load(f)
            assert type(merges) in {list, dict}, f"Error: Expected type of merges is either list or dict, but provided {type(merges)}."
            if type(merges) is list:
                self.merges = {}
                for i, pair in enumerate(merges):
                    self.merges[pair] = i
            else:
                self.merges = merges

    def save_files(self, folder_path: str):
        """
        Serialize 
            - Vocab
            - Merges

        Args:
            folder_path (str): A path to the folder where we want to save vocabulary and list of merges. 
        """
        # Serialize vocab
        with open(Path(folder_path) / "vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        # Serialize merges
        with open(Path(folder_path) / "merges.pkl", "wb") as f:
            pickle.dump(self.merges, f)
        
    @abstractmethod
    def train(self):
        """
        Train / Tokenize tokenizer
        """
        pass
    
    @abstractmethod
    def encode(self, text: str):
        """
        Encode a full text string to a list of token IDs.
        """
        pass
    
    @abstractmethod
    def encode_iterable(self, iterable: Iterator[str]):
        """
        Streamingly encode an iterable of text chunks into token IDs.
        """
        pass

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs into a UTF-8 string.

        Args:
            ids (list[int]): Token ID sequence.

        Returns:
            str: Decoded string.
        """
        byte_seq = b"".join([self.vocab[i] for i in ids])
        text = byte_seq.decode("utf-8", errors="replace")
        return text