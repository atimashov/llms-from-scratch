import regex as re
from time import perf_counter
from typing import BinaryIO, Iterator
from multiprocessing import Pool
from pathlib import Path
import pickle
import numpy as np
import os
from tqdm import tqdm
from .base import Tokenizer

_MERGES = None
_TOKENS2ID = None
_PAT = None
_FILE_PATH = None

# atexit.register(_print_final_worker_stats)

def _init_global_vars(merges, tokens2id, pattern, file_path = None):
    # NOTE: to avoid pickling the whole "self" it is necessary to process outside of the class
    # it makes run 5x faster and saves memory consumption 4x.
    global _MERGES, _TOKENS2ID, _PAT, _FILE_PATH
    _MERGES = merges
    _TOKENS2ID = tokens2id
    _PAT = pattern
    _FILE_PATH = file_path

def _encode_pretoken(pretoken: str, merges: dict, tokens2id: dict):
    b = pretoken.encode("utf-8")
    tokens = [b[i:i+1] for i in range(len(b))]
    while True:
        best_rank = float("inf")
        # find best adjacent pair
        i = 0
        while i < len(tokens) - 1:
            curr_rank = merges.get((tokens[i], tokens[i + 1]), None)
            if curr_rank is not None and curr_rank <= best_rank:
                if curr_rank < best_rank:
                    curr_indices = []
                best_rank = curr_rank
                curr_indices.append(i)
                if i + 2 < len(tokens) and tokens[i] == tokens[i + 1] == tokens[i + 2]:
                    i += 1
            i += 1
        if  best_rank == float('inf'):
            break
        for i in reversed(curr_indices): # to avoid shifting indices
            tokens[i:i+2] = [tokens[i] + tokens[i + 1]]
    return [tokens2id[t] for t in tokens]

def _encode_doc(is_st: bool, doc: str):
    if is_st:
        return [_TOKENS2ID[doc.encode("utf-8")]]
    tokens_ids = []
    for m in _PAT.finditer(doc):
        pretoken = m.group(0)
        tokens_ids.extend(_encode_pretoken(pretoken, _MERGES, _TOKENS2ID))
    # _log_worker_memory("NEW")
    return tokens_ids

def _encode_doc_new(is_st: bool, init_pos: int, doc_size = int):
    with open(_FILE_PATH, "rb") as f:
        f.seek(init_pos)
        doc_bytes = f.read(doc_size)

        if is_st:
            return [_TOKENS2ID[doc_bytes]]
        tokens_ids = []
        for m in _PAT.finditer(doc_bytes.decode("utf-8")):
            pretoken = m.group(0)
            tokens_ids.extend(_encode_pretoken(pretoken, _MERGES, _TOKENS2ID))
    return tokens_ids

class BPETokenizer(Tokenizer):
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

    # def find_docs_boundaries(self, file_path, desired_num_chunks = 4800):
    #     st_pat = b"(" + b"|".join(re.escape(tok.encode("utf-8")) for tok in self.special_tokens) + b")" # TODO: move it to init
    #     pat = re.compile(st_pat)
        
    #     docs_meta = []

    #     with open(file_path, "rb") as f:
    #         # find chunk boundaries
    #         boundaries = self.find_chunk_boundaries(f, desired_num_chunks)
    #         chunks_meta = [(start, end - start) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
    #         # find docs and special tokens boundaries
    #         for init_pos, chunk_size in chunks_meta:
    #             f.seek(init_pos)  # Start at boundary guess
    #             chunk_bytes = f.read(chunk_size)
    #             chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
                
    #             pos = 0
    #             for m in pat.finditer(chunk_str):
    #                 start, end = m.span()
    #                 if pos < start:
    #                     docs_meta.append((False, init_pos + pos, start - pos))
    #                 docs_meta.append((True, init_pos + start, end - start))
    #                 pos = end
    #             if pos < len(chunk_str):
    #                 docs_meta.append((False, init_pos + pos, len(chunk_str) - pos))
    #     return docs


    def find_docs_boundaries(self, file_path, desired_num_chunks = 4800):        
        docs_meta = []

        with open(file_path, "rb") as f:
            # find chunk boundaries
            boundaries = self.find_chunk_boundaries(f, desired_num_chunks)
            chunks_meta = [(start, end - start) for start, end in zip(boundaries[:-1], boundaries[1:])]
            
            for st_bytes in [tok.encode("utf-8") for tok in self.special_tokens]:
                frontier_meta = []
                for init_pos, chunk_size in chunks_meta:
                    f.seek(init_pos)
                    chunk_bytes = f.read(chunk_size)
                    start = 0
                    while True:
                        idx = chunk_bytes.find(st_bytes, start)
                        if idx == -1:
                            frontier_meta.append((init_pos + start, len(chunk_bytes) - start))
                            break
                        docs_meta.append((True, init_pos + idx, len(st_bytes)))
                        if idx > start:
                            frontier_meta.append((init_pos + start, idx - start))
                        start = idx + len(st_bytes)
                chunks_meta = frontier_meta.copy()

            for init_pos, chunk_size in chunks_meta:
                docs_meta.append((False, init_pos, chunk_size))
        return sorted(docs_meta, key = lambda x: x[1])



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

        # run pretokenizer for chunks in parallel # NOTE: probably it makes sense to adjust num_processes (to match real cores) in Pool?
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
            pairs_pos: dict[(bytes, bytes), dict[int, list[int]]]
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
        self.cnt_pretokens[pretoken_idx] = tuple(new_tokens), occur
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
        self.merges[best_pair] = self.cur_vsize # TODO: probably -256 or so
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
        loop = tqdm(range(self.vocab_size - self.cur_vsize), leave = True, desc = "Tokenizer training")
        for _ in loop:
            best_pair, _ = self.train_step()
            loop.set_postfix(curr_vocab_size=f"{self.cur_vsize}/{self.vocab_size}")
        print(f"⏱️ Merges: time={perf_counter() - merge_start:.2f}s") 
        
    def encode_prev(self, text: str, lazy_out_path: str | None = None, num_processes: int = 24) -> list[int] | None:
        """
        Encode a full text string to a list of token IDs.

        Args:
            text (str): Input text to tokenize.
            num_processes (int): Parallelism level for pretoken encoding.

        Returns:
            list[int]: Token IDs after applying merges or None if lazy writes are performed.
        """
        chunk_size = num_processes * 400
        # compile special tokens pattern
        if self.special_tokens:
            st_pat = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
            comp_pat = re.compile(st_pat)
            docs = self.iter_split_bytes(text, comp_pat)
        else:
            docs = [(True, False, text)] 
        
        # iterate over documents
        if not hasattr(self, "vocab_size") and hasattr(self, "vocab"):
            self.vocab_size = len(self.vocab)
        dtype = np.uint16 if self.vocab_size <= 65536 else np.uint32
        # open file for writing if we do it lazily
        fout = None
        if lazy_out_path is not None:
            fout_path = os.path.join(lazy_out_path, "tokens.bin")
            fout = open(fout_path, "ab")
        with Pool(num_processes, initializer=_init_global_vars, initargs=(self.merges, self.tokens2id, re.compile(self.PAT))) as p:
            ids_encoded = []
            docs_chunk = []
            for _, special_token, doc in docs:
                if doc == "":
                    continue
                docs_chunk.append((special_token, doc))
                if len(docs_chunk) >= chunk_size:
                    # run 
                    pretokens_encoded = p.starmap(_encode_doc, docs_chunk)
                    # merge results
                    for ids_pretoken in pretokens_encoded:
                        ids_encoded.extend(ids_pretoken)
                    docs_chunk = []
                    if lazy_out_path is not None:
                        fout.write(np.asarray(ids_encoded, dtype=dtype).tobytes())
                        ids_encoded = []
                    
            # run Pool last time
            if len(docs_chunk) > 0:
                pretokens_encoded = p.starmap(_encode_doc, docs_chunk)
                # merge results
                for ids_pretoken in pretokens_encoded:
                    ids_encoded.extend(ids_pretoken)
                docs_chunk = []
                if lazy_out_path is not None:
                    fout.write(np.asarray(ids_encoded, dtype=dtype).tobytes())
                    print(f"Final file size is {os.path.getsize(os.path.join(lazy_out_path, 'tokens.bin'))}")
                    ids_encoded = []
            if fout:
                fout.close()
        return ids_encoded if len(ids_encoded) > 0 else None

    def encode(self, file_path: str, lazy_out_path: str | None = None, num_processes: int = 24) -> list[int] | None:
        """
        Encode a full text string to a list of token IDs.

        Args:
            text (str): Input text to tokenize.
            num_processes (int): Parallelism level for pretoken encoding.

        Returns:
            list[int]: Token IDs after applying merges or None if lazy writes are performed.
        """

        docs_meta = self.find_docs_boundaries(file_path, desired_num_chunks = 4800)
        print("-->", len(docs_meta))

        # iterate over documents
        dtype = np.uint16 if self.vocab_size <= 65536 else np.uint32

        with Pool(num_processes, initializer=_init_global_vars, initargs=(self.merges, self.tokens2id, re.compile(self.PAT), file_path)) as p:
            chunk_size = num_processes * 400
            ids_encoded = []
            
            loop = tqdm(range(0, len(docs_meta), chunk_size), leave = True, desc = "Encoding", unit="chunks")
            for i in loop:
                docs_meta_chunk = docs_meta[i: i + chunk_size]
                # run 
                pretokens_encoded = p.starmap(_encode_doc_new, docs_meta_chunk)
                # merge results
                for ids_pretoken in pretokens_encoded:
                    ids_encoded.extend(ids_pretoken)
                loop.set_postfix(tokens_encoded=len(ids_encoded))
        return ids_encoded if len(ids_encoded) > 0 else None


    
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Streamingly encode an iterable of text chunks into token IDs.
        Currently does not support parallelization,.
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
                else:
                    for is_tail, is_match, match in self.iter_split_bytes(doc, re.compile(self.PAT)):
                        if not is_match:
                            continue
                        elif is_end and is_tail:
                            tail = match
                        else:
                            for encoded_id in _encode_pretoken(match, self.merges, self.tokens2id):
                                yield encoded_id
        if tail != "":
            for encoded_id in _encode_pretoken(tail, self.merges, self.tokens2id):
                yield encoded_id