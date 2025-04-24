import os, psutil
import json
from collections import defaultdict, OrderedDict
from random import random, shuffle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info

from numpy.random import choice
from tqdm import tqdm
from datetime import datetime
from time import sleep, perf_counter
import gc
from multiprocessing import Manager
from multiprocessing import current_process

class StanfordSentiment(Dataset):
    def __init__(
            self, context = 5, path='datasets/stanfordSentimentTreebank', repeat = 30
    ):
        # TODO: discard rare tokens, experiment with uploading all data to GPU
        self.context = context
        self.path = path
        self.repeat = repeat
        # iterate over sentences
        self._init_sentences()
        # initialize tokens
        self._init_tokens()   
        # initialize probabilities to discard tokens
        self._init_discard_tokens()
        # process sentences
        self._process_sentences() 
    

    def _init_sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return
        
        # initialize all sentences
        sentences = []
        with open(self.path + "/datasetSentences.txt", "r") as f:
            for line in f:
                splitted = line.strip().split()[1:]
                sentences += [[w.lower() for w in splitted]]
        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences]) # NOTE: why not list?
        self._cumsentlen = np.cumsum(self._sentlengths)

    def _init_tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return

        tokens = dict()
        tokenfreq = defaultdict(int)
        tokenfreq_list = []
        revtokens = []
        wordcount = 0
        idx = 0

        for sentence in self._sentences:
            for w in sentence:
                wordcount += 1
                tokenfreq[w] += 1
                if w not in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    idx += 1
                    tokenfreq_list += [1]
                else:
                    tokenfreq_list[tokens[w]] +=1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        tokenfreq_list.append(1)
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._tokenfreq_list = np.array(tokenfreq_list)
        self._wordcount = wordcount
        self._revtokens = revtokens

    def _init_discard_tokens(self):
        if hasattr(self, "_discard_tokens") and self._tokens:
            return
        threshold = 1e-5 * self._wordcount

        nTokens = len(self._tokens)
        discard_tokens = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            discard_tokens[i] = max(0, 1 - np.sqrt(threshold / self._tokenfreq[w]))
        self._discard_tokens = discard_tokens

    def _process_sentences(self):
        if hasattr(self, "_processed_sentences") and self._allsentences:
            return self._allsentences
        discard_tokens = self._discard_tokens
        tokens = self._tokens
        processed_sentences = []
        # NOTE: I expect here different words being discarded for each iteration
        for _ in range(self.repeat):
            processed_sentences += [[w for w in s if random() >= discard_tokens[tokens[w]]] for s in self._sentences]
        # NOTE: it is necessary to ensure it is context to predict
        self._processed_sentences = [s for s in processed_sentences if len(s) > self.context]
        # create matching between overall idx and actual position
        self.idx2position = dict()
        idx = 0
        for i_s, s in enumerate(self._processed_sentences):
            # NOTE: due to the small length of not related sentences, I will predict only preceeding words (it might make sense to add "after" context)
            for i_w in range(len(s)):
                self.idx2position[idx] = (i_s, i_w)
                idx += 1   
        self._center_word_count = idx
    
    def __len__(self):
        return self._center_word_count
    
    def __getitem__(self, idx):
        # load center word and contexts
        # NOTE 2: in the original paper it is +-C, I implement just preceeding words
        i_s, i_w = self.idx2position[idx]
        center_word = self._processed_sentences[i_s][i_w]
        center_id = self._tokens[center_word]
        # NOTE: add random context lengths?
        context_words = [self._processed_sentences[i_s][i_w + i] for i in range(-min(self.context, i_w), 0)]
        context_words += [self._processed_sentences[i_s][i_w + i] for i in range(self.context - len(context_words))]
        context_ids = [self._tokens[word] for word in context_words]
        return center_id, torch.tensor(context_ids)

class GloveDataSet(Dataset):
    """
    NOTE: use batch sizes only that are parts of .pt shapes
    """
    def __init__(self, files_path = 'corpus_tokens_wiki2018/torch_tensors', files_type = '.pt', shared_mem = None):
        super().__init__()
        if shared_mem:
            self.shared_mem = shared_mem
        self.files_path = files_path
        self.total_rows = 0
        # TODO: create proper randomness (seed)
        # create random mapping
        self.chunk_to_len = {}
        for i_chunk, pt_name in enumerate(sorted(os.listdir(files_path))):
            if not pt_name.startswith("cooc_indices_pairs"):
                continue
            tmp_pt = torch.load(os.path.join(files_path, pt_name))
            self.total_rows += tmp_pt.shape[0]
            self.chunk_to_len[i_chunk] = tmp_pt.shape[0]
        self.chunks = len(self.chunk_to_len)
        chunks_id = [i for i in range(self.chunks)]
        shuffle(chunks_id)

        self._chunk_id_to_random = {idx:random_idx for idx, random_idx in enumerate(chunks_id)}


    def _get_chunk_numbers(self, idx):
        cumsum = 0
        for chunk_pos in range(self.chunks):
            chunk_id = self._chunk_id_to_random[chunk_pos]
            pairs_num = self.chunk_to_len[chunk_id]
            if cumsum <= idx < cumsum + pairs_num:
                return chunk_id, idx - cumsum
            cumsum += pairs_num

    def _update_curr_chunk(self, req_chunk, worker_info = None, idx = 0):
        is_global = worker_info is None
        # current chunk is correct (either in global or local memory)
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] _update_curr_chunk 1: {req_chunk} | is_global = {is_global} | idx = {idx}")
        if hasattr(self, 'curr_chunk_number') and self.curr_chunk_number == req_chunk:
            return
        
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] _update_curr_chunk 2: {req_chunk} | is_global = {is_global} | idx = {idx}")
        file_name_pairs = os.path.join(self.files_path, f"cooc_indices_pairs_{req_chunk:04d}.pt")
        file_name_cooc = os.path.join(self.files_path, f"cooc_values_{req_chunk:04d}.pt")
        cache = {
            'pairs_tensor': torch.load(file_name_pairs),
            'cooc_tensor': torch.load(file_name_cooc),
            'shuffled_indices': [i for i in range(self.chunk_to_len[req_chunk])],
        }
        shuffle(cache['shuffled_indices'])

        if is_global:
            self.global_cache = cache
        else:
            self.worker_cache = cache
        self.curr_chunk_number = req_chunk
                # update memory
        self.get_worker_mem_mb_by_worker_id(get_worker_info())
        gc.collect()
        

    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
        worker_info = get_worker_info()
        is_global = worker_info is None # global cache or per-worker cache
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] worker_info")
        # print(worker_info)
        # print()

        # get chunk id and row position inside list of shuffled indices of this chunk
        req_chunk, req_position = self._get_chunk_numbers(idx)
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] req_chunk & req_position", req_chunk, req_position)
        # print()

        # update current chunks and shuffled indices (if necessary)
        self._update_curr_chunk(req_chunk, worker_info, idx)
        if is_global: # global cache (num_workers = 0)
            cache = self.global_cache
        else: # per-worker cache
            cache = self.worker_cache

        row_idx = cache['shuffled_indices'][req_position]
        return cache['pairs_tensor'][row_idx], cache['cooc_tensor'][row_idx]

    def get_worker_mem_mb_by_worker_id(self, worker_info):
        if not hasattr(self, 'shared_mem'):
            return
        worker_id = worker_info.id if worker_info is not None else 'main'
        pid = os.getpid()
        curr = psutil.Process(pid).memory_info().rss / 1024**2  # Resident memory in MB
        self.shared_mem['curr'][worker_id] = curr
        prev_peak = self.shared_mem['peak'].get(worker_id, 0)
        self.shared_mem['peak'][worker_id] = max(curr, prev_peak)


def test():
    # data = StanfordSentiment()
    # data_loader = DataLoader(
    #     data, batch_size = 4, shuffle = True, num_workers = 6, drop_last=True, pin_memory = True
    # )
    # loop = tqdm(data_loader, leave = True)
    
    # for batch_idx, (center_ids, context_ids) in enumerate(loop):
    #     print(type(center_ids), type(context_ids))
    #     loop.set_postfix(imgs_shape=center_ids.shape, lables_shape = context_ids.shape)
    
    # create profiler

    # manager = Manager()
    # shared_mem = manager.dict({
    #     'curr': manager.dict(),
    #     'peak': manager.dict()
    # })
    
    shared_mem = None
    # run Dataset
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    data = GloveDataSet(files_path = 'corpus_tokens_wiki2018/torch_tensors_50K', shared_mem=shared_mem)
    data_loader = DataLoader(
        data, batch_size = 400, shuffle = False, num_workers = 0, drop_last=False, pin_memory = False
    )

    # run Loop
    loop = tqdm(data_loader, leave = True)
    for batch_idx, (_, _) in enumerate(loop):
        # if batch_idx % 1000 == 0:
        #     data.get_worker_mem_mb_by_worker_id(None)
        #     curr_total = sum(data.shared_mem['curr'].values())
        #     peak_total = sum(data.shared_mem['peak'].values())
        #     curr_main = data.shared_mem['curr']['main']
        #     peak_main = data.shared_mem['peak']['main']
        #     curr_1 = data.shared_mem['curr'].get(1, 0)
        #     peak_1 = data.shared_mem['peak'].get(1, 0)
        #     num_workers = len(data.shared_mem['curr'])
        #     gc.collect()
        #     loop.set_postfix(OrderedDict([
        #             ('workers', num_workers), 
        #             ('curr', f"{curr_total:.1f}MB"), 
        #             ('peak', f"{peak_total:.1f}MB"), 
        #             ('curr_main', f"{curr_main:.1f}MB"), 
        #             ('peak_main', f"{peak_main:.1f}MB"), 
        #             ('curr1', f"{curr_1:.1f}MB"), 
        #             ('peak1', f"{peak_1:.1f}MB")
        #         ])

        # )
        pass
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    test()