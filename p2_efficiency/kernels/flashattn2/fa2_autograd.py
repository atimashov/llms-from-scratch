import torch
import triton
from einops import rearrange, einsum

from .fa2_kernels import flashattn_fwd, flashattn_bcwd

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, q_tile=128, k_tile=128, num_warps=4, num_stages=2):
        """
        We don't split over  dimension 'd' on tiles
        """
        # Get dimension
        B, H, N_QUERIES, D_MODEL = Q.shape
        _, H_kv, N_KEYS, _ = K.shape

        # Aserts first
        assert D_MODEL == K.shape[-1], "'d' dimension mismatch"
        assert N_KEYS == V.shape[-2], "Sequence dimension (for K, V) mismatch"
        assert H_kv == V.shape[-3], "Number of heads (for K, V) mismatch"
        assert H % H_kv == 0, "Number of Q heads should be divisible by number of KV heads"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.dtype == K.dtype == V.dtype, "Expected that Q, k, V tensors have the same dtype"

        # Initialize empty result tensor, logsumexps 
        O = torch.empty((B, H, N_QUERIES, D_MODEL), device = Q.device, dtype=Q.dtype)
        L = torch.empty((B, H, N_QUERIES), device=Q.device, dtype=torch.float32)

        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
        stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
        stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
        stride_ob, stride_oh, stride_os, stride_od = O.stride()
        stride_lb, stride_lh, stride_ls = L.stride()
        
        # Run kernel
        scale = 1 / (D_MODEL ** 0.5)
        grid = (B, H, triton.cdiv(N_QUERIES, q_tile))

        flashattn_fwd[grid](
            Q_ptr = Q, K_ptr = K, V_ptr = V, O_ptr = O, L_ptr = L,
            stride_qb = stride_qb, stride_qh = stride_qh, stride_qs = stride_qs, stride_qd = stride_qd,
            stride_kb = stride_kb, stride_kh = stride_kh, stride_ks = stride_ks, stride_kd = stride_kd,
            stride_vb = stride_vb, stride_vh = stride_vh, stride_vs = stride_vs, stride_vd = stride_vd,
            stride_ob = stride_ob, stride_oh = stride_oh, stride_os = stride_os, stride_od = stride_od,
            stride_lb = stride_lb, stride_lh = stride_lh, stride_ls = stride_ls,
            N_QUERIES = N_QUERIES, N_KEYS = N_KEYS, N_HEADS = H, N_HEADS_KV = H_kv,
            scale = scale,
            D_MODEL = D_MODEL, Q_TILE_SIZE = q_tile, K_TILE_SIZE = k_tile, is_causal = is_causal,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Cache vars necessary for backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.Q_TILE_SIZE = q_tile
        ctx.K_TILE_SIZE = k_tile
        ctx.NUM_WARPS = num_warps
        ctx.NUM_STAGES = num_stages

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # Get dimension
        B, H, N_QUERIES, D_MODEL = Q.shape
        _, H_kv, N_KEYS, _ = K.shape

        # Get strides
        stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
        stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
        stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
        stride_ob, stride_oh, stride_os, stride_od = O.stride()
        stride_dob, stride_doh, stride_dos, stride_dod = dO.stride()
        stride_lb, stride_lh, stride_ls = L.stride()

        # Tile sizes and causality
        B_q = ctx.Q_TILE_SIZE
        B_kv = ctx.K_TILE_SIZE
        is_causal = ctx.is_causal
        
        # Init outputs
        dQ = torch.empty_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # Run kernel
        scale = 1 / (D_MODEL ** 0.5)
        grid = (B, H, triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE))
        flashattn_bcwd[grid](
            Q_ptr = Q, K_ptr = K, V_ptr = V, dQ_ptr = dQ, dK_ptr = dK, dV_ptr = dV,
            O_ptr = O, dO_ptr = dO, L_ptr = L,
            stride_qb = stride_qb, stride_qh = stride_qh, stride_qs = stride_qs, stride_qd = stride_qd,
            stride_kb = stride_kb, stride_kh = stride_kh, stride_ks = stride_ks, stride_kd = stride_kd,
            stride_vb = stride_vb, stride_vh = stride_vh, stride_vs = stride_vs, stride_vd = stride_vd,
            stride_ob = stride_ob, stride_oh = stride_oh, stride_os = stride_os, stride_od = stride_od,
            stride_dob = stride_dob, stride_doh = stride_doh, stride_dos = stride_dos, stride_dod = stride_dod,
            stride_lb = stride_lb, stride_lh = stride_lh, stride_ls = stride_ls,
            N_QUERIES = N_QUERIES, N_KEYS = N_KEYS, N_HEADS = H, N_HEADS_KV = H_kv,
            scale = scale,
            D_MODEL = D_MODEL, Q_TILE_SIZE = B_q, K_TILE_SIZE = B_kv,
            is_causal = is_causal,
            num_warps=ctx.NUM_WARPS,
            num_stages=ctx.NUM_STAGES,
        )

        return dQ, dK, dV, None, None, None, None, None