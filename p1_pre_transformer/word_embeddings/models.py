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
    
if __name__ == '__main__':
    pass

