import regex as re
from time import perf_counter
from datetime import datetime
from typing import BinaryIO, Iterator
import os
from multiprocessing import Pool
from pathlib import Path
import argparse
import pickle


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer operating on raw UTF-8 bytes.

    This implementation:
    - Builds the vocabulary from raw input using a BPE merge process
    - Operates at the byte level (instead of indices)
    - Supports streaming and multiprocessing for encoding
    - Supports decoding
    - Respects special tokens (which are never split)
    - Stores the vocabulary as: dict[int → bytes]

    Args:
        input_path (str): Path to input training corpus.
        vocab_size (int): Total maximum vocabulary size.
        special_tokens (list[str] | None): Tokens to preserve as-is (e.g. <|endoftext|>).
    
    Attributes (additional):
        PAT (str): regex Raw string pattern.
        cur_vsize (int): Current vocabulary size, it is required during training. 
        vocab (dict[int, bytes]): ID-to-token mapping.
        tokens2id (dict[bytes, int]): Reverse token lookup.
        merges (list[tuple[bytes, bytes]]): BPE merge rules (in order).
    """
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str] | None):
        self.input_path = input_path

        self.special_tokens = special_tokens
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # init Vocab
        self.vocab = {i: st.encode("utf-8") for i, st in enumerate(special_tokens)}
        self.tokens2id = {st.encode("utf-8"):i  for i, st in enumerate(special_tokens)}
        self.cur_vsize = len(self.vocab)
        for i in range(256):
            self.vocab[self.cur_vsize + i] = bytes([i])
            self.tokens2id[bytes([i])] = self.cur_vsize + i
        self.cur_vsize = len(self.vocab)
        self.vocab_size = vocab_size

        # init merges
        self.merges = []

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

    def iter_split_bytes(self, data: str, pattern: re.Pattern | None):
        """
        Split input text by a regex pattern, yielding flags for special token matches and end position.

        Yields:
            (is_end: bool, is_match: bool, chunk: str)
        """
        # create dummy iterator
        if pattern is None:
            yield True, False, data
        # apply pattern
        else:
            pos = 0
            for m in pattern.finditer(data):
                start, end = m.span()
                if pos < start:
                    yield False, False, data[pos:start]
                is_end = end == len(data)
                yield is_end, True, m.group()
                pos = end
            if pos < len(data):
                yield True, False, data[pos:]

    def pretokenize_chunk(self, start: int, end: int):
        """
        Pre-tokenize a chunk of the input corpus between byte offsets [start, end).

        Reads the chunk from disk, splits it by special tokens, and further each fragment
        is represented as tuple of bytes. Each resulting pre-token (fragment) is counted.

        Args:
            start (int): Byte offset where the chunk begins.
            end (int): Byte offset where the chunk ends.

        Returns:
            dict[tuple[bytes], int]: A dictionary mapping byte sequences (as tuples of 1-byte strings)
            to their frequency counts within the chunk.
        """
        with open(self.input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

        # compile pattern to split chunk (TODO: can I not to split by somehow introduce pattern to make re.finditer?)
        st_pat = "|".join(map(re.escape, self.special_tokens))
        comp_pat = re.compile(st_pat)

        # iterate over splitted chunk
        cnt_pretokens = dict() # NOTE: will Cython - based Counter improve speed?
        my_iter = self.iter_split_bytes(chunk, comp_pat)
        for step in my_iter:
            _, st, fragm = step
            if st:
                continue
            for pretoken in re.finditer(self.PAT, fragm):
                b = pretoken.group().encode("utf-8")
                utf8_bytes = tuple(b[i:i+1] for i in range(len(b)))

                cnt_pretokens[utf8_bytes] = cnt_pretokens.get(utf8_bytes, 0) + 1
        return cnt_pretokens

    def pretokenize(self, num_processes = 24):
        """
        Pre-tokenize the entire corpus in parallel using byte chunking.

        The file is divided into chunks using `find_chunk_boundaries`, ensuring that special tokens
        are not split across chunks. Each chunk is processed in parallel using `pretokenize_chunk`.

        The final result is a frequency list of all UTF-8 byte sequences (pre-tokens), stored in:
            self.cnt_pretokens: list[tuple[tuple[bytes], int]]

        Args:
            num_processes (int): Number of worker processes to use.
        """
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_processes)

            params = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                params.append((start, end))

        # run pretokenizer for chunks in parallel # NOTE: probably it makes sense to adjust num_processes in Pool?
        with Pool(num_processes) as p:
            counters = p.starmap(self.pretokenize_chunk, params)
            
        cnt_pretokens = counters[0]
        for cnt_pretokens_step in counters[1:]:
           for pretokens in cnt_pretokens_step:
                cnt_pretokens[pretokens] = cnt_pretokens.get(pretokens, 0) + cnt_pretokens_step[pretokens] 

        self.cnt_pretokens = [(pretokens_tuple, cnt) for pretokens_tuple, cnt in cnt_pretokens.items()]

    def init_pairs(self):
        """
        Initialize 
            pairs counter (dict[(bytes, bytes), int]):   pair -> number of occurences
            pairs positions (dict[(bytes, bytes), dict[int, list[int]]]): pair -> idx of pretoken -> list of start indices inside pretoken

        Returns:
            pairs_cnt: dict[(bytes, bytes), int]
            pairs_pos: dict[int, list[int]]
        """
        self.pairs_cnt, self.pairs_pos = dict(), dict()
        for i_pretoken, (pretokens_tuple, occur) in enumerate(self.cnt_pretokens):
            prev_pair = None # to avoid duplicating the same pair (e.g. rrrrrrrr)
            for i_pos in range(len(pretokens_tuple) - 1):
                pair = pretokens_tuple[i_pos:(i_pos + 2)]
                # update pairs counter
                self.pairs_cnt[pair] = self.pairs_cnt.get(pair, 0) + occur
                                
                # update pairs positions
                if pair == prev_pair:  # to avoid duplicating the same pair (e.g. rrrrrrrr)
                    prev_pair = None
                    continue
                if pair not in self.pairs_pos:
                    self.pairs_pos[pair] = {}
                if i_pretoken not in self.pairs_pos[pair]:
                    self.pairs_pos[pair][i_pretoken] = []
                self.pairs_pos[pair][i_pretoken].append(i_pos)
                prev_pair = pair  # to avoid duplicating the same pair (e.g. rrrrrrrr)
        return self.pairs_cnt, self.pairs_pos

    def update_pretoken(self, pair: tuple, pretoken_idx: int):
        """
        Apply merge to a specific pretoken, modifying token list in-place.
        Args:
            pair (tuple(bytes, bytes)): pair of bytes to merge
            pretoken_idx: index of pre-token to merge
        
        Returns:
            tuple(bytes): Previous version of pre-tokens
        """
        positions = self.pairs_pos[pair][pretoken_idx]
        pretokens, occur = self.cnt_pretokens[pretoken_idx]
                 
        prev, new_tokens = 0, []
        for curr in positions:
            new_tokens.extend(list(pretokens[prev:curr]))
            new_tokens.append(pair[0] + pair[1])
            prev = curr + 2
        if prev < len(pretokens):
            new_tokens.extend(list(pretokens[prev:]))
        self.cnt_pretokens[pretoken_idx] = tuple(new_tokens), occur # NOTE: do I need to move to tuple?
        return pretokens

    def update_pairs(self, pretoken_idx: int, pretokens_old: tuple):
        """
        Update pair counts and pair positions after modifying a specific pretoken.

        Args:
            pretoken_idx (int): Index of the modified pretoken.
            pretokens_old (tuple): Original pretoken before modification.
        """
        pretokens, occur = self.cnt_pretokens[pretoken_idx]
        for pos in range(len(pretokens_old) - 1):
            pair = pretokens_old[pos:(pos+2)]
            if pretoken_idx in self.pairs_pos[pair]:
                del self.pairs_pos[pair][pretoken_idx]

            self.pairs_cnt[pair] -= occur
            if self.pairs_cnt[pair] == 0:
                del self.pairs_cnt[pair]

        prev_pair = None # to avoid duplicating the same pair (e.g. rrrrrrrr)
        for pos in range(len(pretokens) - 1):
            pair = tuple(pretokens[pos:(pos+2)])
            self.pairs_cnt[pair] = self.pairs_cnt.get(pair, 0) + occur
            if pair == prev_pair: # to avoid duplicating the same pair (e.g. rrrrrrrr)
                prev_pair = None
                continue
            if pair not in self.pairs_pos:
                self.pairs_pos[pair] = dict()
            if pretoken_idx not in self.pairs_pos[pair]:
                self.pairs_pos[pair][pretoken_idx] = []
            self.pairs_pos[pair][pretoken_idx].append(pos)
            
            prev_pair = pair # to avoid duplicating the same pair (e.g. rrrrrrrr)

    def train_step(self):
        """
        Run one merge step of BPE:
            1. Find most frequent pair
            2. Merge across pretoken list
            3. Update vocab and merge list

        Returns:
            tuple(bytes): pair of bytes that we merged 
            int: number of appearences of this pair in the corpus
        """
        best_pair, best_pair_cnt = max(self.pairs_cnt.items(), key=lambda x: (x[1], x[0]))
        # update merges
        self.merges.append(best_pair)
        # update vocabulary
        self.vocab[self.cur_vsize] = best_pair[0] + best_pair[1]
        self.tokens2id[best_pair[0] + best_pair[1]] = self.cur_vsize
        self.cur_vsize += 1
        # modify counter & co-occurence
        i_pretokens = list(self.pairs_pos[best_pair].keys())
        for pretoken_idx in i_pretokens:
            pretokens_old = self.update_pretoken(best_pair, pretoken_idx)
            self.update_pairs(pretoken_idx, pretokens_old)
        return best_pair, best_pair_cnt


    def train(self, num_processes: int = 24):
        """
        Train BPE merges until vocab_size is reached.

        Args:
            num_processes (int): Number of processes for parallel pretokenization.
        """
        if not hasattr(self, "cnt_pretokens"):
            t = perf_counter()
            self.pretokenize(num_processes)
            print(f"⏱️ Pretokenized: time={perf_counter() - t:.2f}s")
        
        merge_start = perf_counter()
        # step 1: init counter one time
        self.init_pairs()
        t = perf_counter()
        while self.cur_vsize < self.vocab_size:
            best_pair, _ = self.train_step()
        print(f"⏱️ Merges: time={perf_counter() - merge_start:.2f}s") 

    def from_files(self, vocab_path: str, merges_path: str):
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
        # deserialize merges
        with open(merges_path, "rb") as f:
            self.merges = pickle.load(f)


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
    
    def encode_pretoken(self, pretoken: str):
        """
        Encode a single pre-token into BPE IDs using current merge rules.

        Args:
            pretoken (str): A Unicode string token.

        Returns:
            list[int]: Token IDs after applying merges.
        """
        # create initial list of bytes
        b = pretoken.encode("utf-8")
        tokens = [b[i:i+1] for i in range(len(b))]
        for token1, token2 in self.merges:
            # create indices of occurences
            indices = []
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == token1 and tokens[i + 1] == token2:
                    tokens[i:i+2] = [token1 + token2]
                i += 1
        return [self.tokens2id[b] for b in tokens]


    def encode(self, text: str, num_processes: int = 24) -> list[int | bytes]:
        """
        Encode a full text string to a list of token IDs.

        Args:
            text (str): Input text to tokenize.
            num_processes (int): Parallelism level for pretoken encoding.

        Returns:
            list[int]: Token IDs after applying merges.
        """
        # compile special tokens pattern
        if self.special_tokens:
            st_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            comp_pat = re.compile(st_pat)
            docs = self.iter_split_bytes(text, comp_pat)
        else:
            docs = [(True, False, text)] 

        # iterate over documents
        ids_encoded = []
        for _, special_token, doc in docs:
            if doc == "":
                continue
            if special_token:
                ids_encoded.append(self.tokens2id[doc.encode("utf-8")])
            else:
                # split doc on pretoken
                pretokens = re.findall(self.PAT, doc) # NOTE: OR re.finditer?
                # encode each pretoken in parallel
                with Pool(min(num_processes, len(pretokens))) as p:
                    pretokens_encoded = p.map(self.encode_pretoken, pretokens)
                
                # merge results
                for ids_pretoken in pretokens_encoded:
                    ids_encoded.extend(ids_pretoken)
        return ids_encoded
    
    def encode_iterable(self, iterable: Iterator[str], num_processes: int = 1) -> Iterator[int]:
        """
        Streamingly encode an iterable of text chunks into token IDs.
        Trade-off: 
            If we use multiprocessing, it will be faster, but we will use more RAM.

        Args:
            iterable (Iterator[str]): Stream of text input chunks.
            num_processes (int): Use multiprocessing if > 1. Otherwise process tokens in-place.

        Yields:
            int: Token IDs one-by-one.
        """
        #  special tokens pattern
        if self.special_tokens:
            st_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            comp_pat = re.compile(st_pat)
        else:
            comp_pat = None
        
        # iterate over chunks
        tail = ""
        for chunk in iterable:
            chunk = tail + chunk
            docs_iter = self.iter_split_bytes(chunk, comp_pat)
            
            for is_end, special_token, doc in docs_iter:
                if special_token:
                    if is_end:
                        tail = ""
                    yield self.tokens2id[doc.encode("utf-8")]
                # run Pool in parallel (if we are OK to allocate more memory)
                elif num_processes > 1:
                        pretokens = re.findall(self.PAT, doc)
                        if is_end:
                            tail = pretokens[-1]
                            pretokens = pretokens[:-1]
                        if pretokens:
                            with Pool(min(num_processes, len(pretokens))) as p:
                                pretokens_encoded = p.map(self.encode_pretoken, pretokens)
                # run main streaming (without multiprocessing)
                else:
                    for is_tail, is_match, match in self.iter_split_bytes(doc, re.compile(self.PAT)):
                        if not is_match:
                            continue
                        elif is_end and is_tail:
                            tail = match
                        else:
                            for encoded_id in self.encode_pretoken(match):
                                yield encoded_id
        if tail != "":
            encoded_ids = self.encode_pretoken(tail)
            for encoded_id in encoded_ids:
                yield encoded_id

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE tokenizer trainer")
    parser.add_argument("--input-path", type = str, default=str(Path.cwd().parents[2] / "data" / "TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--vocab-size", type = int, default = 10_000)
    parser.add_argument("--num-processes", type = int, default = 24)
    parser.add_argument("--special-tokens", nargs = "+", type = str, default = ["<|endoftext|>"], help = "List of special tokens")
    args = parser.parse_args()

    bpe = BPETokenizer(input_path = args.input_path, vocab_size = args.vocab_size, special_tokens = args.special_tokens)


    t = perf_counter()
    bpe.pretokenize()
    print(f"⏱️ Pretokenized: time={perf_counter() - t:.2f}s") 

    bpe.train()

    # simple text input
    test_text = "Sasha Likes Capybaras<|endoftext|> capybaras like sasha<|endoftext|>"

    t = perf_counter()
    ids = bpe.encode(test_text)
    print("Encoded IDs:", ids)
    print(f"⏱️ Encoded: time={perf_counter() - t:.2f}s") 

    t = perf_counter()
    decoded = bpe.decode(ids)
    print("Decoded Text:", decoded)
    print(f"⏱️ Decoded: time={perf_counter() - t:.2f}s") 

    print("\n--- Streaming encoding test ---")
    text_iter = iter([
        "Sasha Likes Capybaras<|endoftext|> capybaras li",
        "ke sasha<|endoftext|>"
    ])
    t = perf_counter()
    iter = list(bpe.encode_iterable(text_iter))
    print("Streamed IDs:")
    for streamed_id in iter:
        print(streamed_id)
    print(f"⏱️ Encoded iter: time={perf_counter() - t:.2f}s")
