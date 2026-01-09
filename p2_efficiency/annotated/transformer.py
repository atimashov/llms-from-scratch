import torch
from torch import nn
import torch.cuda.nvtx as nvtx
from einops import rearrange, einsum

from p1_core.utils import softmax
from p1_core.layers import Embedding, Linear, RMSNorm, LayerNorm, GatedFFN, FFN, MultiHeadSelfAttention

    
# class AnnotatedMHSA(MultiHeadSelfAttention):
# """
#     d_model: int Dimensionality of the Transformer block inputs
#     num_heads: int Number of heads to use in multi-head self-attention.
#     """
#     def __init__(
#         self, d_model: int, num_heads: int, num_heads_kv: int = 1, theta: float = 10000.0, 
#         context_length = 10000, init_type: str = 'xavier', clip_w: float = 3.0,
#         device: torch.device | None = None, dtype: torch.dtype | None = None, kv_cache: bool = False, block_num: int = 0
#         ):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
#         assert num_heads % num_heads_kv == 0, "'num_heads' should be divisible by 'num_heads_kv'"

#         self.block_num = block_num

#         self.num_heads_q = num_heads # queries
#         self.num_heads_kv = num_heads_kv   # keys & values
#         self.cntx = context_length
#         self.d_model = d_model
#         self.d_qk = d_model // num_heads
#         self.d_v = d_model // num_heads
        
#         # Init Projections
#         self.P_Q = Linear(d_model, self.num_heads_q * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
#         self.P_K = Linear(d_model, self.num_heads_kv * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
#         self.P_V = Linear(d_model, self.num_heads_kv * self.d_v, init_type, clip_w, device = device, dtype=dtype)
#         self.P_O = Linear(self.num_heads_q * self.d_v, d_model, init_type, clip_w, device = device, dtype=dtype)
        
#         # Init RoPE
#         self.rope = RoPE(theta = theta, d_qk = self.d_qk, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
    
#         # Init KV Cache
#         if kv_cache:
#             self.register_buffer(
#                 "cache_k",
#                 torch.empty((1, self.num_heads_kv, self.cntx, self.d_qk), device = device, dtype = dtype),
#                 persistent = False
#             )
#             self.register_buffer(
#                 "cache_v",
#                 torch.empty((1, self.num_heads_kv, self.cntx, self.d_v), device = device, dtype = dtype),
#                 persistent = False
#             )
#             self.start, self.kv_cache_len = 0, 0
#     def forward(self, x: torch.Tensor, is_causal: bool = True, with_rope = True):
#         # Calculate token positions
#         with nvtx.range(f"Attn block {self.block_num}: tokens pos"):
#             if hasattr(self, "kv_cache_len") and self.kv_cache_len > 0:
#                 assert x.shape[-2] == 1, "You don't need to provide the whole input after prefill"
#                 seq_len = min(self.cntx, self.kv_cache_len + 1)
#                 token_positions_q = torch.tensor([seq_len - 1], dtype=torch.long, device=x.device)
#                 token_positions_kv = torch.arange(seq_len, device=x.device)
#             else:
#                 seq_len = x.shape[-2]
#                 token_positions_q = token_positions_kv = torch.arange(seq_len, device=x.device)

#         # Project x to get queries (Q), keys (K) and values
#         with nvtx.range(f"Attn block {self.block_num}: project Q"):
#             q = self.get_proj_q(x, with_rope, token_positions_q)  # batch_size x h x seq_len x d_qk  or batch_size x 1 x d_qk (KV cache)
#         with nvtx.range(f"Attn block {self.block_num}: project K, V"):
#             k, v = self.get_proj_kv(x, with_rope, token_positions_kv)  # batch_size x h x seq_len x d_qk

#         # Calculate scaled MHA
#         with nvtx.range(f"Attn block {self.block_num}: MHA"):
#             scaled_mha = scaled_dot_product_attention(q, k, v, is_causal = is_causal)
#         with nvtx.range(f"Attn block {self.block_num}: concat heads"):
#         scaled_mha = rearrange(scaled_mha, "... h_q seq_len_q d_v -> ... seq_len_q (h_q d_v)")
        
#         # Project an output
#         with nvtx.range(f"Attn block {self.block_num}: final projection"):
#             o = self.P_O(scaled_mha)
#         return o

class AnnotatedTransformerBlock(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer.
    """
    def __init__(
        self, d_model: int, d_ff: int, num_heads: int, activation: str, is_gate: bool, theta: float = 10000.0,
        context_length = 10000, init_type: str = 'xavier', clip_w: float = 3.0, norms: dict | None = None, 
        device: torch.device | None = None, dtype: torch.dtype | None = None, block_num: int = 0
        ):
        super().__init__()
        assert norms.get("before", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("after", None) in {"RMSNorm", "LayerNorm", None}
        assert norms.get("residual", None) in {"RMSNorm", "LayerNorm", None}
        
        self.block_num = block_num

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

        self.attn = MultiHeadSelfAttention(
            d_model = d_model, num_heads = num_heads, theta = theta, context_length = context_length, 
            init_type = init_type, clip_w = clip_w, device = device, dtype = dtype
            )
        ffn = GatedFFN if is_gate else FFN
        self.ffn = ffn(d_model = d_model, d_hidden = d_ff, init_type = init_type, clip_w = clip_w, activation = activation, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        with nvtx.range(f"block {self.block_num}"):
            # apply the first block (Multi Head Self Attention)
            with nvtx.range(f"block {self.block_num}: ln before"):
                ln_before = self.bef_ln1(x)
            with nvtx.range(f"block {self.block_num}: attention"):
                attn = self.attn(ln_before)
            with nvtx.range(f"block {self.block_num}: attn residuals"):
                res = x + self.aft_ln1(attn)
            with nvtx.range(f"block {self.block_num}: attn ln"):
                y = self.res_ln1(res)
            # apply the first block (Feed Forward)
            with nvtx.range(f"block {self.block_num}: ffn"):
                ff = self.ffn(self.bef_ln2(y))
            with nvtx.range(f"block {self.block_num}: ffn residuals"):
                res_ff = y + self.aft_ln2(ff)
            with nvtx.range(f"block {self.block_num}: ffn ln"):
                out = self.res_ln2(res_ff)
        return out
    
class AnnotatedTransformerLM(nn.Module):
    """
    vocab_size: embedding matrix.
    int The size of the vocabulary, necessary for determining the dimensionality of the token
    context_length: int The maximum context length, necessary for determining the dimensionality of
    the position embedding matrix.
    num_layers: int The number of Transformer blocks to use.
    """
    def __init__(
        self, d_model: int, d_ff: int, num_heads: int, activation: str, is_gate: bool, num_layers:int = 6, theta: float = 10000.0, 
        context_length = 256, init_type: str = 'xavier', std_emb: float = 0.02, clip_w: float = 3.0, vocab_size: int = 10_000,
        norms: dict | None = None, weights_tying: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        assert norms.get("final", None) in {"RMSNorm", "LayerNorm", None}
        self.context_length = context_length
        self.token_embeddings = Embedding(num_embeddings = vocab_size, embedding_dim = d_model, std = std_emb, clip_w = clip_w, device = device, dtype = dtype)
        
        self.layers = nn.Sequential(
            *[AnnotatedTransformerBlock(
                d_model = d_model, d_ff = d_ff, num_heads = num_heads, activation = activation, is_gate = is_gate, theta = theta,
                context_length = context_length, init_type = init_type, clip_w = clip_w, norms = norms, device = device, dtype = dtype
                ) for _ in range(num_layers)]
        )

        if norms.get("final", None) is None:
            self.ln_final = nn.Identity()
        else:
            self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype) if norms["final"] == "RMSNorm" else LayerNorm(d_model=d_model, device=device, dtype=dtype)
        
        tied_weight = self.token_embeddings.emb if weights_tying else None
        self.lm_head = Linear(d_model, vocab_size, init_type, clip_w, tied_weight = tied_weight, device=device,dtype=dtype)
    
    def forward(self, token_ids, prob: bool = False, tau: float = 1.0):
        with nvtx.range("token emb"):
            x = self.token_embeddings(token_ids)
        with nvtx.range("transformer blocks"):
            x = self.layers(x)
        with nvtx.range("final ln"):
            x = self.ln_final(x)
        with nvtx.range("vocab projection"):
            logits = self.lm_head(x)
        # if self.wt:
        #     logits /= self.d_model ** 0.5
        if not prob:
            return logits
        probs = softmax(logits, dim= -1, tau = tau)
        return probs






