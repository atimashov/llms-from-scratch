from torch import einsum
import torch
from torch import nn
from einops import rearrange, einsum
from nonlinearity import SiLU
from utils import scaled_dot_product_attention, softmax
from pos_enc import RoPE
from norms import RMSNorm

class Linear(nn.Module):
    """
    Linear Layer

    This implementation:
    - Applies Linear Transformation
    - Does not support bias since modern implementations of transformers does not use it (I might add it later)
    - Initialize weight with Normal distribution with parameters: 
        - mu = 0; std = 2 / (in_features + out_features)
        - clip them (-3 * std, 3 * std)
    
    Args:
        in_features: final dimension of the input
        out_features: final dimension of the output
        device: Device to store the parameters on
        dtype: Data type of the parameters 
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # initialize data and create linear transformation
        std = 2 / (in_features + out_features)
        data = torch.empty(out_features, in_features, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(data)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.device == self.W.device
        Y = einsum(X, self.W, "... d_in, d_out d_in -> ... d_out")
        return Y

        
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # initialize data and create embedding table
        data = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.emb = nn.Parameter(data)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]       

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W1 = Linear(d_model, d_hidden, device = device, dtype=dtype)
        self.W2 = Linear(d_hidden, d_model, device = device, dtype=dtype)
        self.W3 = Linear(d_model, d_hidden, device = device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, X):
        return self.W2(self.silu(self.W1(X)) * self.W3(X))

class MultiHeadSelfAttention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs
    num_heads: int Number of heads to use in multi-head self-attention.
    TODO: check dtype/device correctness; add documentation; work on reproducibility (seed)
    """
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, context_length = 10000, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k, self.d_v = d_model // num_heads, d_model // num_heads
        # init matrices
        std = 2 / (num_heads * self.d_k + d_model)
        data = torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_Q = nn.Parameter(data)
        data = torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_K = nn.Parameter(data)
        std = 2 / (num_heads * self.d_v + d_model)
        data = torch.empty(num_heads * self.d_v, d_model, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_V = nn.Parameter(data)
        data = torch.empty(d_model, num_heads * self.d_v, dtype=dtype, device=device)
        nn.init.trunc_normal_(data, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.P_O = nn.Parameter(data)
        # init RoPE
        self.rope = RoPE(theta = theta, d_k = self.d_k, max_seq_len= context_length, device=device, dtype = dtype)
        self.device = device

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, token_positions = None):
        # project x to get queries, keys and values
        Q = einsum(x, self.P_Q, "... d, hd_k d -> ... hd_k") # unplug
        Q = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        K = einsum(x, self.P_K, "... d, hd_k d -> ... hd_k")
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
        V = einsum(x, self.P_V, "... d, hd_v d -> ... hd_v")
        V = rearrange(V, "...  seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads)
        # apply RoPE
        if with_rope:
            Q = self.rope(Q)
            K = self.rope(K)
        # create mask
        if is_masked:
            mask = ~torch.triu(torch.full((Q.shape[-2], K.shape[-2]), True, device = self.device), diagonal=1)
        else:
            mask = None
        # calculate scaled attention
        scaled_mh_att = scaled_dot_product_attention(Q, K, V, mask)
        scaled_mh_att = rearrange(scaled_mh_att, "... h seq_len d_v -> ... seq_len (h d_v)")
        # project on output
        O = einsum(scaled_mh_att, self.P_O, "... seq_len hd_v, d hd_v -> ... seq_len d")
        return O
    
class Transformer(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0, context_length = 10000, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model = d_model, num_heads = num_heads, theta = theta, context_length = context_length, device = device, dtype = dtype)
        self.ffn = SwiGLUFFN(d_model = d_model, d_hidden = d_ff, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        # apply the first block (Multi Head Self Attention)
        y = x + self.attn(self.ln1(x))
        # apply the first block (Feed Forward)
        return y + self.ffn(self.ln2(y))
    
class TransformerLM(nn.Module):
    """
    vocab_size: embedding matrix.
    int The size of the vocabulary, necessary for determining the dimensionality of the token
    context_length: int The maximum context length, necessary for determining the dimensionality of
    the position embedding matrix.
    num_layers: int The number of Transformer blocks to use.
    """
    def __init__(self, d_model: int, d_ff: int, num_heads: int, num_layers:int = 6, theta: float = 10000.0, context_length = 256, vocab_size: int = 10_000, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, device = device, dtype = dtype)
        self.layers = nn.Sequential(
            *[Transformer(d_model = d_model, num_heads = num_heads, d_ff = d_ff, theta = theta, context_length = context_length, device = device, dtype = dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device,dtype=dtype)
    
    def forward(self, token_ids, prob = False):
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if not prob:
            return logits
        probs = softmax(logits, dim= -1)
        return probs






