import torch
from torch import nn
from einops import rearrange, einsum

from p1_core.utils import softmax
from p1_core.layers import Embedding, Linear, RMSNorm, LayerNorm, GatedFFN, FFN, MultiHeadSelfAttention, MultiHeadLatentAttention

def get_attention(
    attn_params:dict, d_model: int, theta: float, context_length: int, init_type: str,
    clip_w: float, device: torch.device | None, dtype: torch.dtype | None, kv_cache: bool = False
    ):
    attn_type = attn_params["type"]
    num_heads = attn_params["num_heads"]
    num_heads_kv = attn_params.get("num_heads_kv", num_heads)
    d_latent = attn_params.get("d_latent", None)
    
    if attn_type == "mla":
        attn = MultiHeadLatentAttention(
            d_model = d_model, d_latent = d_latent, num_heads = num_heads,  theta = theta, 
            context_length = context_length, init_type = init_type, clip_w = clip_w,
            device = device, dtype = dtype, kv_cache = kv_cache
            )
    else:
        attn = MultiHeadSelfAttention(
            d_model = d_model, num_heads = num_heads, num_heads_kv = num_heads_kv, theta = theta,
            context_length = context_length, init_type = init_type, clip_w = clip_w,
            device = device, dtype = dtype, kv_cache = kv_cache
            )
    return attn
    
class TransformerBlock(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(
        self, d_model: int, d_ff: int, attn_params: dict, activation: str, is_gate: bool, theta: float = 10000.0,
        context_length = 10000, init_type: str = 'xavier', clip_w: float = 3.0, norms: dict | None = None, 
        device: torch.device | None = None, dtype: torch.dtype | None = None, kv_cache: bool = False
        ):
        super().__init__()
        assert norms.get("before", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("after", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("residual", None) in {"RMSNorm", "LayerNorm", None}
        
        if norms.get("before", None) is None:
            self.bef_ln1 = nn.Identity()
            self.bef_ln2 = nn.Identity()
        else:
            self.bef_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["before"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.bef_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["before"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        if norms.get("after", None) is None:
            self.aft_ln1 = nn.Identity()
            self.aft_ln2 = nn.Identity()
        else:
            self.aft_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["after"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.aft_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["after"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        if norms.get("residual", None) is None:
            self.res_ln1 = nn.Identity()
            self.res_ln2 = nn.Identity()
        else:
            self.res_ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["residual"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype) 
            self.res_ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["residual"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)

        self.attn = get_attention(attn_params = attn_params, d_model = d_model, theta = theta, 
            context_length = context_length, init_type = init_type, clip_w = clip_w, 
            device = device, dtype = dtype, kv_cache = kv_cache
        )
        ffn = GatedFFN if is_gate else FFN
        self.ffn = ffn(d_model = d_model, d_hidden = d_ff, init_type = init_type, clip_w = clip_w, activation = activation, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        # apply the first block (Multi Head Self Attention)
        attn = self.attn(self.bef_ln1(x))
        res = x + self.aft_ln1(attn)
        y = self.res_ln1(res)
        # apply the first block (Feed Forward)
        ff = self.ffn(self.bef_ln2(y))
        res_ff = y + self.aft_ln2(ff)
        out = self.res_ln2(res_ff)
        return out
    
class TransformerLM(nn.Module):
    """
    vocab_size: embedding matrix.
    int The size of the vocabulary, necessary for determining the dimensionality of the token
    context_length: int The maximum context length, necessary for determining the dimensionality of
    the position embedding matrix.
    num_layers: int The number of Transformer blocks to use.
    """
    def __init__(
        self, d_model: int, d_ff: int, attn_params: dict, activation: str, is_gate: bool, num_layers:int = 6, theta: float = 10000.0, 
        context_length = 256, init_type: str = 'xavier', std_emb: float = 0.02, clip_w: float = 3.0, vocab_size: int = 10_000,
        norms: dict | None = None, weights_tying: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None,
        kv_cache: bool = False
        ):
        super().__init__()
        assert norms.get("final", None) in {"RMSNorm", "LayerNorm", None}
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, std = std_emb, clip_w = clip_w, device = device, dtype = dtype)
        
        self.layers = nn.Sequential(
            *[TransformerBlock(
                d_model = d_model, d_ff = d_ff, attn_params = attn_params, activation = activation, is_gate = is_gate, theta = theta,
                context_length = context_length, init_type = init_type, clip_w = clip_w, norms = norms, device = device, dtype = dtype, kv_cache = kv_cache
                ) for _ in range(num_layers)]
        )

        if norms.get("final", None) is None:
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["final"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        
        tied_weight = self.token_embeddings.emb if weights_tying else None
        self.lm_head = Linear(d_model, vocab_size, init_type, clip_w, tied_weight = tied_weight, device=device,dtype=dtype)
    
    def forward(self, token_ids, prob: bool = False, tau: float = 1.0):
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        # if self.wt:
        #     logits /= self.d_model ** 0.5
        if not prob:
            return logits
        probs = softmax(logits, dim= -1, tau = tau)
        return probs






