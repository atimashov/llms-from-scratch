from torch import nn
import torch
import numpy as np

class SkipGram(nn.Module):
    def __init__(self, emb_size = 300, vocab_size = -1, neg_sample = None, freq = None, device = 'cpu'):
        super(SkipGram, self).__init__()
        self.embs_center = nn.Embedding(vocab_size, emb_size)
        self.embs_context = nn.Embedding(vocab_size, emb_size) 
        self.vocab_size = vocab_size
        self.neg_sample = neg_sample
        self.device = device
        if neg_sample:
            self.all_indices = np.arange(vocab_size)
            # TODO: modify frequences
            self.sample_prob = (freq)**0.75/ ((freq)**0.75).sum()


    def forward(self, center_ids, context_ids):
        # print("----> ", center_ids.shape, context_ids.shape)
        B, C = context_ids.shape
        embs_center = self.embs_center(center_ids).unsqueeze(2) # BxDx1
        # print("*** embs_center ->", embs_center.shape)
        embs_context = self.embs_center(context_ids.reshape(-1)).reshape((B, C, -1)) # BxCxD
        # print("*** embs_context ->", embs_context.shape)

        if self.neg_sample: # NOTE: currently computationally not efficient
            # NOTE: super not efficient, just for the example
            # batch_indices = torch.arange(B).unsqueeze(1).expand(-1, self.neg_sample)
            subsets_neg = []
            for i in range(B):
                # print("<><><>", self.all_indices.shape, self.sample_prob.shape)
                indices = torch.from_numpy(np.random.choice(self.all_indices, size=self.neg_sample, replace=False, p=self.sample_prob))
                indices = indices.to(self.device)
                subsets_neg.append(self.embs_context(indices))
            embs_negatives = torch.stack(subsets_neg, dim = 0) # B x K x D
            # print('negatives ids shape: ', embs_negatives.shape)
            # embs_negatives = self.embs_center(negatives_ids.reshape(-1)).reshape((B, self.neg_sample, -1)) # BxKxD
            # calculate cosine distances
            cos_positive = torch.bmm(embs_context, embs_center)
            cos_negative = torch.bmm(embs_negatives, embs_center)
        else:
            cos_positive = None
            cos_negative = None
        return cos_positive, cos_negative    

class GloVe(nn.Module):
    def __init__(self, vocab_size = 400_000, emb_size = 300, x_max = 100, _alpha = 0.75):
        super(GloVe, self).__init__()
        self.x_max = x_max
        self._alpha = _alpha
        # init embeddings and its weights
        scale = 0.5 / emb_size
        self.w_center = nn.Embedding(vocab_size, emb_size)
        self.w_context = nn.Embedding(vocab_size, emb_size)
        self.b_center =  nn.Embedding(vocab_size, 1)
        self.b_context = nn.Embedding(vocab_size, 1)
        with torch.no_grad():
            self.w_center.weight.uniform_(-scale, scale)
            self.w_context.weight.uniform_(-scale, scale)
            self.b_center.weight.zero_()
            self.b_context.weight.zero_()

    def forward(self, ids, X, epsilon = 1e-8):
        dot = (self.w_center(ids[:,0]) * self.w_context(ids[:,1])).sum(axis = 1, keepdim = True)
        bias = (self.b_center(ids[:,0]) + self.b_context(ids[:,1]))
        weight = self._weight_f(X)
        delta = dot + bias - torch.log(X + epsilon)
        return weight, delta

    def _weight_f(self, x):
        mask = x < self.x_max
        out = torch.ones_like(x)
        out[mask] = (x[mask] / self.x_max) ** self._alpha
        return out




if __name__ == '__main__':
    input = torch.LongTensor([[1, 2], [4, 5], [4, 3], [2, 9]])
    X = torch.randn(4, 1)
    model = GloVe(vocab_size = 1000)
    weight, delta = model(input, X)
    print(input.shape, X.shape, weight.shape, delta.shape)

