from torch import einsum
import torch
from torch import nn
from einops import rearrange, einsum

from utils import scaled_dot_product_attention, softmax
from layers import Embedding, Linear, RMSNorm, LayerNorm, SwiGLUFFN, MultiHeadSelfAttention

    
class Transformer(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0, context_length = 10000, norms: dict | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert norms.get("before", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("after", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("residual", None) in {"RMSNorm", "LayerNorm", None}
        
        if norms["before"] is None:
            self.bef_ln1 = nn.Identity()
            self.bef_ln2 = nn.Identity()
        else:
            self.bef_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["before"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.bef_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["before"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        if norms["after"] is None:
            self.aft_ln1 = nn.Identity()
            self.aft_ln2 = nn.Identity()
        else:
            self.aft_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["after"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.aft_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["after"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        if norms["residual"] is None:
            self.res_ln1 = nn.Identity()
            self.res_ln2 = nn.Identity()
        else:
            self.res_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["residual"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.res_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["residual"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        
        self.attn = MultiHeadSelfAttention(d_model = d_model, num_heads = num_heads, theta = theta, context_length = context_length, device = device, dtype = dtype)
        self.ffn = SwiGLUFFN(d_model = d_model, d_hidden = d_ff, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        # apply the first block (Multi Head Self Attention)
        y = self.res_ln1(x + self.aft_ln1(self.attn(self.bef_ln1(x))))
        # apply the first block (Feed Forward)
        return self.res_ln2(y + self.aft_ln2(self.ffn(self.bef_ln2(y))))
    
class TransformerLM(nn.Module):
    """
    vocab_size: embedding matrix.
    int The size of the vocabulary, necessary for determining the dimensionality of the token
    context_length: int The maximum context length, necessary for determining the dimensionality of
    the position embedding matrix.
    num_layers: int The number of Transformer blocks to use.
    """
    def __init__(self, d_model: int, d_ff: int, num_heads: int, num_layers:int = 6, theta: float = 10000.0, context_length = 256, vocab_size: int = 10_000, norms: dict | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        assert norms.get("final", None) in {"RMSNorm", "LayerNorm", None}
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, device = device, dtype = dtype)
        self.layers = nn.Sequential(
            *[Transformer(d_model = d_model, num_heads = num_heads, d_ff = d_ff, theta = theta, context_length = context_length, norms = norms, device = device, dtype = dtype) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        if norms["final"] is None:
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["final"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
 
        self.lm_head = Linear(d_model, vocab_size, device=device,dtype=dtype)
    
    def forward(self, token_ids, prob: bool = False, tau: float = 1.0):
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if not prob:
            return logits
        probs = softmax(logits, dim= -1, tau = tau)
        return probs






