from einops import rearrange, einsum
import torch
from torch import nn
from .pos_enc import RoPE
from .core import Linear
from p1_core.utils import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    """
    Q:  (batch_size, ..., seq_len_q, d_qk) # seq_len_q = 1 for KV Cache
    K:  (batch_size, ..., seq_len_kv, d_qk)
    V:  (batch_size, ..., seq_len_kv, d_v)
    """
    d_qk = K.shape[-1]
    H_q, H_kv = Q.shape[1], K.shape[1]
    
    # Grouped-Query Attention reshape
    if H_q > H_kv and H_kv > 1: # GQA
        Q = rearrange(Q, "... (h_kv r) seq_len_q d_qk -> ... h_kv r seq_len_q d_qk", h_kv = H_kv)
        K = rearrange(K, "... h_kv seq_len_kv d_qk -> ... h_kv 1 seq_len_kv d_qk")
        V = rearrange(V, "... h_kv seq_len_kv d_v -> ... h_kv 1 seq_len_kv d_v")
    
    # Compute scores
    scores = einsum(Q, K, "... seq_len_q d_qk, ... seq_len_kv d_qk -> ... seq_len_q seq_len_kv")
    scores = scores.clamp(min = -80, max=80.0) # NOTE: might be better after rescaling
    
    # Compute masking
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Compute weights
    weights = softmax(scores / (d_qk ** 0.5), dim = -1)
    
    # Compute attention
    attn = einsum(weights, V, "... seq_len_q seq_len_kv, ... seq_len_kv d_v -> ... seq_len_q d_v")

    # Rearrange to (batch_size, seq_len, h_q * d_v)
    if H_q > H_kv and H_kv > 1: # GQA
        attn = rearrange(attn, "... h_kv r seq_len_q d_v -> ... seq_len_q (h_kv r d_v)")
    else: # MHA or MQA
        attn = rearrange(attn, "... h_q seq_len_q d_v -> ... seq_len_q (h_q d_v)")
    return attn

class MultiHeadSelfAttention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs
    num_heads: int Number of heads to use in multi-head self-attention.
    TODO: check dtype/device correctness; add documentation;
    """
    def __init__(
        self, d_model: int, num_heads: int, kv_heads_ratio: int = 1, theta: float = 10000.0, context_length = 10000, max_len = 10000, kv_cache: bool = False,
        init_type: str = 'xavier', clip_w: float = 3.0, device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % kv_heads_ratio == 0, "'num_heads' should be divisible by 'kv_heads_ratio'"
        self.num_heads_q = num_heads # queries
        self.num_heads_kv = num_heads // kv_heads_ratio   # keys & values
        self.d_model = d_model
        self.d_qk = d_model // num_heads
        self.d_v = d_model // num_heads
        self.device = device
        
        # Init Projections
        self.P_Q = Linear(d_model, self.num_heads_q * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
        self.P_K = Linear(d_model, self.num_heads_kv * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
        self.P_V = Linear(d_model, self.num_heads_kv * self.d_v, init_type, clip_w, device = device, dtype=dtype)
        self.P_O = Linear(self.num_heads_q * self.d_v, d_model, init_type, clip_w, device = device, dtype=dtype)
        
        # Init RoPE
        self.rope = RoPE(theta = theta, d_qk = self.d_qk, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
        
        # Init KV Cache
        if kv_cache: # TODO: check if dtype is correct for AMP
            # NOTE: for now it works only for batch_size = 1
            with torch.no_grad():
                self.kv_cache = {
                    "K": torch.zeros((1, self.num_heads_kv, max_len, self.d_qk), device = device, dtype = dtype),
                    "V": torch.zeros((1, self.num_heads_kv, max_len, self.d_v), device = device, dtype = dtype),
                }
    
    def get_proj_q(self, x):
        """
        Compute query (Q) projection.

        if KV cache is enabled, it will return just projection on the last token.
        Otherwise - projection of all sequence.

        Args:
            x:  (batch_size, seq_len, d_model)

        Returns:
            Q: (batch_size, num_heads, seq_len, d_qk) or (batch_size, num_heads, 1, d_qk)
        """
        Q = self.P_Q(x[:,-1:,:] if hasattr(self, "kv_cache") else x)
        return rearrange(Q, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_q)

    def get_proj_kv(self, x: torch.Tensor, name: str):
        """
        Compute and (optionally) update K/V cache.

        If 'kv_cache' is turned off (training, full seq inference), projects the entire input.
        If 'kv_cache' is turned on (autoregressive inference), updates only the last token in the cache
        and returns the prefix up to seq_len.

        Args:
            x:  (batch_size, seq_len, d_model)
            name: 'K' or 'V'

        Returns:
            out: (batch_size, h, seq_len, d_proj)
        """
        assert name in {"K", "V"}, f"Name can be only 'K' or 'V' but provided '{name}'"
        # dynamically select projection
        proj = getattr(self, f"P_{name}")

        if hasattr(self, "kv_cache"):
            batch_size, seq_len, _ = x.shape
            assert seq_len <= self.kv_cache[name].shape[2], "Exceeded max_len of KV cache"
            assert batch_size == 1, f"Currently support batch_size = 1, but provided {batch_size}"

            proj_new = proj(x[:,-1:,:]) # batch_size x 1 x d_model
            proj_new = rearrange(proj_new, "... seq_len (h d_proj) -> ... h seq_len d_proj", h = self.num_heads_kv) # batch_size x num_heads x 1 x d_model
            self.kv_cache[name][:, :, seq_len-1:seq_len,:] = proj_new
            out = self.kv_cache[name][:,:,:seq_len,:]
        else:
            out = proj(x)  # batch_size x seq_len x d_model
            out = rearrange(out, "... seq_len (h d_proj) -> ... h seq_len d_proj", h = self.num_heads_kv)
        return out

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, token_positions = None):
        # Project x to get queries (Q), keys (K) and values
        Q = self.get_proj_q(x)  # batch_size x h x seq_len x d_qk  or batch_size x 1 x d_qk (KV cache)
        K = self.get_proj_kv(x, "K")  # batch_size x h x seq_len x d_qk
        V = self.get_proj_kv(x, "V")  # batch_size x h x seq_len x d_v
        
        # Apply RoPE
        if with_rope and self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        
        # Create mask
        if is_masked:
            mask = ~torch.triu(torch.full((Q.shape[-2], K.shape[-2]), True, device = self.device), diagonal=1)
        else:
            mask = None
        
        # Calculate scaled DPA
        scaled_mh_att = scaled_dot_product_attention(Q, K, V, mask)
        
        # Project an output
        O = self.P_O(scaled_mh_att)
        return O

class MultiHeadLatentAttention(nn.Module):
    """
    d_model: int Dimensionality of the Transformer block inputs
    num_heads: int Number of heads to use in multi-head self-attention.
    TODO: check dtype/device correctness; add documentation;
    """
    def __init__(
        self, d_model: int, d_latent: int, num_heads: int, theta: float = 10000.0, context_length = 10000, max_len = 10000, kv_cache: bool = False,
        init_type: str = 'xavier', clip_w: float = 3.0, device: torch.device | None = None, dtype: torch.dtype | None = None
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_latent = d_latent        
        self.d_h = d_model // num_heads
        self.d_hr = self.d_h // 2

        self.device = device
        
        # Init Projections
        self.P_DKV = Linear(d_model, d_latent, init_type, clip_w, device = device, dtype=dtype)
        self.P_UK = Linear(d_latent, self.num_heads * self.d_h, init_type, clip_w, device = device, dtype=dtype)
        self.P_UV = Linear(d_latent, self.num_heads * self.d_h, init_type, clip_w, device = device, dtype=dtype)

        self.P_DQ = Linear(d_model, d_latent, init_type, clip_w, device = device, dtype=dtype)
        self.P_UQ = Linear(d_latent, self.num_heads * self.d_h, init_type, clip_w, device = device, dtype=dtype)

        self.P_QR = Linear(d_latent, self.d_hr * self.num_heads, init_type, clip_w, device = device, dtype=dtype)
        self.P_KR = Linear(d_model, self.d_hr, init_type, clip_w, device = device, dtype=dtype)

        self.P_O = Linear(self.num_heads * self.d_h, d_model, init_type, clip_w, device = device, dtype=dtype)
        
        # Init RoPE
        self.rope = RoPE(theta = theta, d_k = self.d_qk, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
        
        # Init KV Cache
        if kv_cache: # TODO: check if dtype is correct for AMP
            # NOTE: for now it works only for batch_size = 1
            with torch.no_grad():
                self.kv_cache = {
                    "K": torch.zeros((1, self.num_heads_kv, max_len, self.d_qk), device = device, dtype = dtype),
                    "V": torch.zeros((1, self.num_heads_kv, max_len, self.d_v), device = device, dtype = dtype),
                }
    
    def get_proj_q(self, x):
        """
        Compute query (Q) projection.

        if KV cache is enabled, it will return just projection on the last token.
        Otherwise - projection of all sequence.

        Args:
            x:  (batch_size, seq_len, d_model)

        Returns:
            Q: (batch_size, num_heads, seq_len, d_qk) or (batch_size, num_heads, 1, d_qk)
        """
        Q = self.P_Q(x[:,-1:,:] if hasattr(self, "kv_cache") else x)
        return rearrange(Q, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_q)

    def get_proj_kv(self, x: torch.Tensor, name: str):
        """
        Compute and (optionally) update K/V cache.

        If 'kv_cache' is turned off (training, full seq inference), projects the entire input.
        If 'kv_cache' is turned on (autoregressive inference), updates only the last token in the cache
        and returns the prefix up to seq_len.

        Args:
            x:  (batch_size, seq_len, d_model)
            name: 'K' or 'V'

        Returns:
            out: (batch_size, h, seq_len, d_proj)
        """
        assert name in {"K", "V"}, f"Name can be only 'K' or 'V' but provided '{name}'"
        # dynamically select projection
        proj = getattr(self, f"P_{name}")

        if hasattr(self, "kv_cache"):
            batch_size, seq_len, _ = x.shape
            assert seq_len <= self.kv_cache[name].shape[2], "Exceeded max_len of KV cache"
            assert batch_size == 1, f"Currently support batch_size = 1, but provided {batch_size}"

            proj_new = proj(x[:,-1:,:]) # batch_size x 1 x d_model
            proj_new = rearrange(proj_new, "... seq_len (h d_proj) -> ... h seq_len d_proj", h = self.num_heads_kv) # batch_size x num_heads x 1 x d_model
            self.kv_cache[name][:, :, seq_len-1:seq_len,:] = proj_new
            out = self.kv_cache[name][:,:,:seq_len,:]
        else:
            out = proj(x)  # batch_size x seq_len x d_model
            out = rearrange(out, "... seq_len (h d_proj) -> ... h seq_len d_proj", h = self.num_heads_kv)
        return out

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, token_positions = None):
        # # Project x to latents
        # c_KV = self.P_DKV(x) # batch_size x seq_len x d_latent
        # c_Q = self.P_DQ(x) # batch_size x seq_len x d_latent

        # # Project latent to heads dimension
        # K, V = self.P_UK(c_KV), self.P_UV(c_KV) # batch_size x seq_len x hn_h
        # K = rearrange(K, "")
        
        # K, V = 
        # Q = self.P_UQ(c_Q)
        
        
        
        
        # Project x to get queries (Q), keys (K) and values
        Q = self.get_proj_q(x)  # batch_size x h x seq_len x d_qk  or batch_size x 1 x d_qk (KV cache)
        K = self.get_proj_kv(x, "K")  # batch_size x h x seq_len x d_qk
        V = self.get_proj_kv(x, "V")  # batch_size x h x seq_len x d_v
        
        # Apply RoPE
        if with_rope and self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        
        # Create mask
        if is_masked:
            mask = ~torch.triu(torch.full((Q.shape[-2], K.shape[-2]), True, device = self.device), diagonal=1)
        else:
            mask = None
        
        # Calculate scaled DPA
        scaled_mh_att = scaled_dot_product_attention(Q, K, V, mask)
        
        # Project an output
        O = self.P_O(scaled_mh_att)
        return O