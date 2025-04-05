import os
import json
from collections import defaultdict
from random import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from numpy.random import choice
from tqdm import tqdm

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

def test():
    data = StanfordSentiment()
    data_loader = DataLoader(
        data, batch_size = 4, shuffle = True, num_workers = 6, drop_last=True, pin_memory = True
    )
    loop = tqdm(data_loader, leave = True)
    
    for batch_idx, (center_ids, context_ids) in enumerate(loop):
        print(type(center_ids), type(context_ids))
        loop.set_postfix(imgs_shape=center_ids.shape, lables_shape = context_ids.shape)

if __name__ == '__main__':
    test()