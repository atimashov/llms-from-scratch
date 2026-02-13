__all__ = [
    "flash_attention",
    "annotated_scaled_dot_product_attention"
]

from einops import einsum
import torch
import torch.cuda.nvtx as nvtx
from p1_core.utils import softmax
from p2_efficiency.kernels.flashattn2 import FlashAttention2


@nvtx.range("flash attention 2")
def flash_attention(Q, K, V, is_causal: bool = True, q_tile=64, k_tile=64, num_warps=4, num_stages=1):
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    return FlashAttention2.apply(Q, K, V, is_causal, q_tile, k_tile, num_warps, num_stages)

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = True):
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
    with nvtx.range("compute attention scores"):
        scores = einsum(Q, K, "... seq_len_q d_qk, ... seq_len_kv d_qk -> ... seq_len_q seq_len_kv")
        scores = scores.clamp(min = -80, max=80.0) # NOTE: might be better after rescaling
    
    # Compute masking
    with nvtx.range("apply mask"):
        if is_causal:
            seq_q, seq_kv = scores.shape[-2:]
            assert seq_q <= seq_kv, f"causal assumes S_Q <= S_K, got S_Q = {seq_q}, S_K = {seq_kv}"
            token_pos_kv = torch.arange(seq_kv, device = scores.device)
            token_pos_q = torch.arange(seq_kv - seq_q, seq_kv, device = scores.device)
            mask = token_pos_q[:, None] >= token_pos_kv[None, :]
            scores = scores.masked_fill(~mask, float('-inf'))
    
    with nvtx.range("computing softmax"):
        # Compute weights
        weights = softmax(scores / (d_qk ** 0.5), dim = -1)
    with nvtx.range("computing values projections and rearranging"):    
        # Compute attention
        attn = einsum(weights, V, "... seq_len_q seq_len_kv, ... seq_len_kv d_v -> ... seq_len_q d_v")
       
        # Rearrange GQA to (B, H, S, d_h)
        if H_q > H_kv and H_kv > 1: # GQA
            attn = rearrange(attn, "... h_kv r seq_len_q d_h -> ... (h_kv r) seq_len_q d_h")
    return attn