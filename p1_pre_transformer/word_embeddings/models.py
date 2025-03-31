from torch import nn

class CBOW(nn.Module):
    def __init__(self, emb_size = 300, vocab_size = -1):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).mean(1) # batch_size x emb_size
        return self.linear(embeddings) 

