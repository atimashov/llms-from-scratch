from einops import rearrange, einsum
import torch
from torch import nn
from typing import List
from .pos_enc import RoPE
from .core import Linear
from p1_core.utils import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None):
    """
    Q:  (batch_size, ..., seq_len_q, d_qk) # seq_len_q = 1 for KV Cache
    K:  (batch_size, ..., seq_len_kv, d_qk)
    V:  (batch_size, ..., seq_len_kv, d_v)
    """
    # print("========Q, K, V=========", Q.shape, K.shape, V.shape)
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
    # print("=== attn before rearrange ===", attn.shape)
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
    """
    def __init__(
        self, d_model: int, num_heads: int, num_heads_kv: int = 1, theta: float = 10000.0, 
        context_length = 10000, max_len = 10000, init_type: str = 'xavier', clip_w: float = 3.0,
        device: torch.device | None = None, dtype: torch.dtype | None = None, kv_cache: bool = False
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_heads_kv == 0, "'num_heads' should be divisible by 'num_heads_kv'"
        self.num_heads_q = num_heads # queries
        self.num_heads_kv = num_heads_kv   # keys & values
        self.max_len = max_len
        self.d_model = d_model
        self.d_qk = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Init Projections
        self.P_Q = Linear(d_model, self.num_heads_q * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
        self.P_K = Linear(d_model, self.num_heads_kv * self.d_qk, init_type, clip_w, device = device, dtype=dtype)
        self.P_V = Linear(d_model, self.num_heads_kv * self.d_v, init_type, clip_w, device = device, dtype=dtype)
        self.P_O = Linear(self.num_heads_q * self.d_v, d_model, init_type, clip_w, device = device, dtype=dtype)
        
        # Init RoPE
        self.rope = RoPE(theta = theta, d_qk = self.d_qk, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
    
        # Init KV Cache
        if kv_cache:
            with torch.no_grad():
                self.kv_cache = {
                    "k": torch.zeros((1, self.num_heads_kv, self.max_len, self.d_qk), device = device, dtype = dtype),
                    "v": torch.zeros((1, self.num_heads_kv, self.max_len, self.d_v), device = device, dtype = dtype),
                }
            self.prefilled = False

    def get_proj_q(self, x: torch.Tensor, with_rope: bool = True, token_positions: torch.Tensor | None = None):
        """
        Compute query (Q) projection.
        if KV cache is enabled, it is expected taht sequence length is 1.

        Args:
            x:  (batch_size, seq_len, d_model)

        Returns:
            Q: (batch_size, num_heads, seq_len, d_qk) or (batch_size, num_heads, 1, d_qk)
        """
        q = self.P_Q(x)
        q = rearrange(q, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_q)
        
        if with_rope and self.rope is not None: # TODO: deal with token_positions
            q = self.rope(q, token_positions = token_positions)
        return q
    
    def get_proj_kv(self, x: torch.Tensor, with_rope: bool = True, token_positions: torch.Tensor | None = None):
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
        if hasattr(self, "kv_cache"):
            batch_size, seq_len, _ = x.shape
            assert seq_len <= self.max_len, "Exceeded max_len of KV cache"
            assert batch_size == 1, f"Currently support batch_size = 1, but provided {batch_size}"
            
            # populate KV Cache (NOTE: for now it works only for batch_size = 1)
            if not self.prefilled:
                k = self.P_K(x)  # batch_size x seq_len x (h d_qk)
                k = rearrange(k, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_kv)
                if with_rope and self.rope is not None:
                    k = self.rope(k, token_positions = token_positions)
                self.kv_cache["k"][:, :, :seq_len,:] = k
            
                v = self.P_V(x)  # batch_size x seq_len x (h d_v)
                v = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads_kv)
                self.kv_cache["v"][:, :, :seq_len,:] = v
                self.curr_kv_len = seq_len
                self.prefilled = True
            else:
                pos = self.curr_kv_len
                assert pos < self.max_len, "KV cache full"
                assert x.shape[-2] == 1, "You don't need to provide the whole input after prefill"

                k_new = self.P_K(x) # batch_size x 1 x (h d_qk)
                k_new = rearrange(k_new, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_kv) # batch_size x num_heads x 1 x d_qk
                if with_rope and self.rope is not None:
                    k_new = self.rope(k_new, token_positions = token_positions)
                self.kv_cache["k"][:, :, pos:pos+1,:] = k_new

                v_new = self.P_V(x) # batch_size x 1 x (h d_v)
                v_new = rearrange(v_new, "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads_kv) # batch_size x num_heads x 1 x d_v
                self.kv_cache["v"][:, :, pos:pos+1,:] = v_new
                
                self.curr_kv_len += 1
                with torch.no_grad():
                    k = self.kv_cache["k"][:,:,:self.curr_kv_len,:]
                    v = self.kv_cache["v"][:,:,:self.curr_kv_len,:]
        else:
            k = self.P_K(x)  # batch_size x seq_len x (h d_qk)
            k = rearrange(k, "... seq_len (h d_qk) -> ... h seq_len d_qk", h = self.num_heads_kv)
            if with_rope and self.rope is not None:
                k = self.rope(k, token_positions = token_positions)

            v = self.P_V(x)  # batch_size x seq_len x (h d_v)
            v = rearrange(v, "... seq_len (h d_v) -> ... h seq_len d_v", h = self.num_heads_kv)
        
        return k, v

    def forward(self, x: torch.Tensor, is_masked: bool = True, with_rope = True, kv_cache = False):
        # calculate token positions
        if hasattr(self, "kv_cache") and self.prefilled:
            assert x.shape[-2] == 1, "You don't need to provide the whole input after prefill"
            token_positions_q = token_positions_kv = torch.tensor([self.curr_kv_len], dtype=torch.long, device=x.device)            
        else:
            seq_len = x.shape[-2]
            token_positions_q = token_positions_kv = torch.arange(seq_len, device=x.device)

        # Project x to get queries (Q), keys (K) and values
        q = self.get_proj_q(x, with_rope, token_positions_q)  # batch_size x h x seq_len x d_qk  or batch_size x 1 x d_qk (KV cache)
        k, v = self.get_proj_kv(x, with_rope, token_positions_kv)  # batch_size x h x seq_len x d_qk
        
        # Create mask
        if is_masked:
            mask = ~torch.triu(torch.full((q.shape[-2], k.shape[-2]), True, device = x.device), diagonal=1)
        else:
            mask = None
        
         # Calculate scaled MHA
        scaled_mha = scaled_dot_product_attention(q, k, v, mask)
        
        # Project an output
        o = self.P_O(scaled_mha)
        return o

class AbsorbedLinear(nn.Module):
    """
    Linear Layer absorbing multiple matrices.
    
    Args:
        - layers (torch.Tensor): linear layers to absorb
        - num_heads (int): number of heads
    """
    def __init__(self, layers: List[torch.Tensor], num_heads: int):
        super().__init__()
        assert 2 <= len(layers) <= 3, f"It is expected not less than 2 and not more than 3 matrices to absorb, but {len(layers)} provided"
        
        if len(layers) == 3:
            d0_out, _ = layers[0].shape
            d1_out, d1_in = layers[1].shape
            _, d2_in = layers[2].shape
            assert d0_out == d1_in and d1_out == d2_in, "Problems with dimensions"
            assert d2_in % num_heads == 0, "Output dimension should be divisible by number of heads"
            with torch.no_grad():
                W0 = layers[1] @ layers[0]
                W1 = layers[2]
        else:
            d0_out, _ = layers[0].shape
            _, d1_in = layers[1].shape
            assert d0_out == d1_in, "Problems with dimensions"
            assert d1_in % num_heads == 0, "Shared intermediate dimension should be divisible by number of heads"
            W0 = layers[0]
            W1 = layers[1]
        # Split along the shared (h*d_h) dim and absorb per-head
        W0 = rearrange(W0, "(h d_h) d_in -> h d_h d_in", h = num_heads)
        W1 = rearrange(W1, "d_out (h d_h) -> h d_out d_h", h = num_heads)

        # Per-head composition: W_head[h] = W1_head[h] @ W0_head[h]
        W = einsum(W1, W0, "h d_out d_h, h d_h d_in  -> h d_out d_in")

        # Store as buffer (no grad)
        self.register_buffer("W", W)  # (h, d_out, d_in)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.device == self.W.device
        return einsum(X, self.W, "... seq_len d_in, h d_out d_in -> ... h seq_len d_out")

class MultiHeadLatentAttention(nn.Module):
    """
    Args:
        d_model (int): Dimensionality of the Transformer block inputs
        d_latent (int): Dimensionality of the latent space  
        num_heads (int): Number of heads to use in Query.
        ...
    """
    def __init__(
        self, d_model: int, d_latent: int, num_heads: int, theta: float = 10000.0, context_length = 10000, max_len = 10000,
        init_type: str = 'xavier', clip_w: float = 3.0, device: torch.device | None = None, dtype: torch.dtype | None = None,
        kv_cache: bool = False
        ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.max_len = max_len
        self.d_model = d_model
        self.d_latent = d_latent        
        self.d_h = d_model // num_heads
        assert self.d_h % 2 == 0, "d_h must be divisible by 2"
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
        self.rope = RoPE(theta = theta, d_qk = self.d_hr, max_seq_len= context_length, device=device, dtype = dtype) if theta is not None else None
    
        if kv_cache:
            # Init KV Cache
            with torch.no_grad():
                self.kv_cache = {
                    "c_KV": torch.zeros((1, 1, self.max_len, self.d_latent), device = device, dtype = dtype),
                    "k_R": torch.zeros((1, 1, self.max_len, self.d_hr), device = device, dtype = dtype),
                }
            self.prefilled = False

    def get_proj_q(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        Compute query (Q) as concat of content and rotary channels.
        if KV cache is enabled, it is expected that sequence length is 1.
        
        Args:
            x:  (batch_size, seq_len, d_model)
        
        Returns:
            q: (B, H, S, d_qc + d_qr)
        """

        # Absorb matrices if KV
        if hasattr(self, "kv_cache") and not hasattr(self, "P_KQ_absorbed"):
            self.P_KQ_absorbed = AbsorbedLinear([self.P_UQ.W, self.P_UK.W.T], self.num_heads)
            self.P_OV_absorbed = AbsorbedLinear([self.P_UV.W, self.P_O.W], self.num_heads)

        # Project x to latents
        c_Q = self.P_DQ(x) # (B x S x d_latent)
    
        # Content part
        if hasattr(self, "kv_cache"):
            q_C = self.P_KQ_absorbed(c_Q) # (batch_size, h, seq_len, d_latent) 
        else:
            q_C = self.P_UQ(c_Q) # (B, S, H * d_h)
            q_C = rearrange(q_C, "... seq_len (h d_h) -> ... h seq_len d_h", h = self.num_heads) # (batch_size, h, seq_len, d_h)

        # RoPE part
        q_R = self.P_QR(c_Q) # (B, S, H * d_hr)
        q_R = rearrange(q_R, "... seq_len (h d_hr) -> ... h seq_len d_hr", h = self.num_heads) # (batch_size, h, seq_len, d_hr)
        if getattr(self, "rope", None) is not None and q_R.size(-1) > 0:
            q_R = self.rope(q_R, token_positions = token_positions) # (batch_size, h, seq_len, d_hr)

        # Concatenate Q-content and Q-RoPE along feature dim
        q = torch.cat((q_C, q_R), dim = -1) # (batch_size, h, seq_len, d_h + d_hr) or (batch_size, h, seq_len, d_latent + d_hr)
        return q

    def get_proj_kv(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        Compute and (optionally) update K/V cache.
        
        Args:
            x:  (batch_size, seq_len, d_model)
            token_positions:

        Returns:
            k / [c_kv, r_K]: (batch_size, h_or_1, seq_len, d_proj)
            v / c_KV: (batch_size, h_or_1, seq_len, d_proj)
        """
        if hasattr(self, "kv_cache"):
            batch_size, seq_len, _ = x.shape
            assert seq_len <= self.max_len, "Exceeded max_len of KV cache"
            assert batch_size == 1, f"Currently support batch_size = 1, but provided {batch_size}"
            
            # 1. Project x to latent space
            c_KV = self.P_DKV(x) # (batch_size, seq_len, d_latent)
            c_KV = c_KV.unsqueeze(1) # (batch_size, 1, seq_len, d_latent)

            # 2. Project x to RoPE part
            k_R = self.P_KR(x) # batch_size x seq_len x d_hr
            if getattr(self, "rope", None) is not None and k_R.size(-1) > 0:
                k_R = self.rope(k_R, token_positions = token_positions) # batch_size x seq_len x d_hr            
            k_R = k_R.unsqueeze(1) # (B, 1, S, d_hr)

            # 3. populate KV Cache
            if not self.prefilled:
                self.kv_cache["c_KV"][:, :, :seq_len, :] = c_KV
                self.kv_cache["k_R"][:, :, :seq_len, :] = k_R
                                
                # Change flag
                self.prefilled = True

                # Update current sequence length
                self.curr_kv_len = seq_len
            else:
                pos = self.curr_kv_len
                assert pos < self.max_len, "KV cache full"
                assert x.shape[-2] == 1, "You don't need to provide the whole input after prefill"

                self.kv_cache["c_KV"][:, :, pos:pos+1, :] = c_KV
                self.kv_cache["k_R"][:, :, pos:pos+1, :] = k_R

                # Update current sequence length
                self.curr_kv_len += 1

            c_KV = self.kv_cache["c_KV"][:, :, :self.curr_kv_len, :] # (batch_size, 1, seq_len, d_latent)
            k_R = self.kv_cache["k_R"][:, :, :self.curr_kv_len, :] # (batch_size, 1, seq_len, d_hr)

            # Concatenate k (latent content and RoPE)
            k = torch.cat((c_KV, k_R), dim = -1) # (batch_size, 1, seq_len, d_latent + d_hr) 
            v_C = c_KV # (batch_size, 1, seq_len, d_latent)
        else:
            # Project x to latents
            c_KV = self.P_DKV(x) # (batch_size, seq_len, d_latent)

            # RoPE part for K
            k_R = self.P_KR(x) # (batch_size, seq_len, d_hr)
            if getattr(self, "rope", None) is not None and k_R.size(-1) > 0:
                k_R = self.rope(k_R, token_positions = token_positions) # (batch_size, seq_len, d_hr)
            
            # Broadcast single-head k_R across heads
            k_R = k_R.unsqueeze(1) # (batch_size, 1, seq_len, d_hr)
            
            # 4. calculate necessary projections
            k_C = self.P_UK(c_KV) # (batch_size, seq_len, h * d_h)
            k_C = rearrange(k_C, "... seq_len (h d_h) -> ... h seq_len d_h", h = self.num_heads) # (batch_size, h, seq_len, d_h)

            k_R = k_R.expand(-1, self.num_heads, -1, -1) # (batch_size, h, seq_len, d_hr)

            # Concatenate k
            k = torch.cat((k_C, k_R), dim = -1) # (batch_size, h, seq_len, d_h + d_hr)

            v_C = self.P_UV(c_KV) # (batch_size, seq_len, h * d_h)
            v_C = rearrange(v_C, "... seq_len (h d_h) -> ... h seq_len d_h", h = self.num_heads) # (batch_size, h, seq_len, d_h)
        return k, v_C

    def forward(self, x: torch.Tensor, is_masked: bool = True):
        # calculate token positions
        if hasattr(self, "kv_cache"):
            assert x.shape[-2] == 1, "You don't need to provide the whole input after prefill"
            token_positions_q = token_positions_kv = torch.tensor([self.curr_kv_len], dtype=torch.long, device=x.device)            
        else:
            seq_len = x.shape[-2]
            token_positions_q = token_positions_kv = torch.arange(seq_len, device=x.device)

        # Project x to get queries (Q), keys (K) and values
        q = self.get_proj_q(x, token_positions_q)  # batch_size x h x seq_len x d_qk
        k, v_C = self.get_proj_kv(x, token_positions_kv)  # batch_size x h x seq_len x d_qk

        # Create mask
        mask = ~torch.triu(torch.full((q.shape[-2], k.shape[-2]), True, device = x.device), diagonal=1) if is_masked else None
        
        # Calculate scaled DPA
        scaled_mha = scaled_dot_product_attention(q, k, v_C, mask)
        # ("=======scaled_mha=========", scaled_mha.shape)
        # Project an output
        final_proj = self.P_OV_absorbed if hasattr(self, "kv_cache") else self.P_O
        # print("=======final_proj=========", final_proj.W.shape)
        O = final_proj(scaled_mha)
        return O