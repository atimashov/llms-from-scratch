import triton
import triton.language as tl

@triton.jit
def flashattn_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    N_QUERIES, N_KEYS, N_HEADS, N_HEADS_KV,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    # Program indices
    b_index = tl.program_id(0) # batch
    h_index = tl.program_id(1) # head
    q_index = tl.program_id(2) # tile id along sequence dimension for Q
    GROUP_SIZE = N_HEADS // N_HEADS_KV

    # Create block pointers using corresponding offsets
    base_q = Q_ptr + b_index * stride_qb + h_index * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base_q,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qs, stride_qd),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_k = K_ptr + b_index * stride_kb + (h_index // GROUP_SIZE) * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base_k,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_v = V_ptr + b_index * stride_vb + (h_index // GROUP_SIZE) * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base_v,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_o = O_ptr + b_index * stride_ob + h_index * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base_o,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_os, stride_od),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_l = L_ptr + b_index * stride_lb + h_index * stride_lh
    L_block_ptr = tl.make_block_ptr(
        base_l,
        shape=(N_QUERIES,),
        strides=(stride_ls, ),
        offsets=(q_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )

    # Initialize buffers for O, L, m to write to
    O = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32) # (Q_TILE_SIZE, D)
    L = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32) # (Q_TILE_SIZE,)
    m = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32) # (Q_TILE_SIZE,)

    # Load the current tile of Q
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D)

    q_start = q_index * Q_TILE_SIZE
    for k_iter in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current tiles of K, V
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # Compute scaled tile score
        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal mask
        if is_causal:
            k_start = k_iter * K_TILE_SIZE
            q_ids = q_start + tl.arange(0, Q_TILE_SIZE)
            k_ids = k_start + tl.arange(0, K_TILE_SIZE)
            mask = (q_ids[:, None] >= k_ids[None, :]) & (k_ids[None, :] < N_KEYS) & (q_ids[:, None] < N_QUERIES)
            S = tl.where(mask, S, S - 1e6)

        # Compute temporary running features
        m_hat = S.max(axis = -1) # (Q_TILE_SIZE, )
        m_new = tl.maximum(m, m_hat) # (Q_TILE_SIZE,)

        # Compute exponent of score and substract max values 
        P_j = tl.exp(S - m_new[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update L
        mult = tl.exp(m - m_new) # (Q_TILE_SIZE,)
        L = L * mult + tl.sum(P_j, axis = -1) # (Q_TILE_SIZE,)

        # Compute running output
        O = O * mult[:, None] + tl.dot(P_j.to(V.dtype), V) # (Q_TILE_SIZE, D)

        # Update running max
        m = m_new # (Q_TILE_SIZE,)

        # Move the pointers to the next tile.
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0)) # (K_TILE_SIZE, D)
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0)) # (K_TILE_SIZE, D)
        
    # Write output to the output and log softmax block pointers
    tl.store(O_block_ptr, (O / L[:, None]).to(Q.dtype), boundary_check=(0,1)) # (Q_TILE_SIZE, D)
    tl.store(L_block_ptr, m + tl.log(L), boundary_check=(0,)) # (Q_TILE_SIZE,)


@triton.jit
def flashattn_bcwd(
    Q_ptr, K_ptr, V_ptr, dQ_ptr, dK_ptr, dV_ptr,
    O_ptr, dO_ptr, L_ptr, D_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    N_QUERIES, N_KEYS, N_HEADS, N_HEADS_KV,
    scale,
    D_MODEL: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    # Program indices
    b_index = tl.program_id(0) # batch
    h_index = tl.program_id(1) # head
    q_index = tl.program_id(2) # tile id along sequence dimension for Q
    GROUP_SIZE = N_HEADS // N_HEADS_KV

    # Create block pointers using corresponding offsets
    base_q = Q_ptr + b_index * stride_qb + h_index * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base_q,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qs, stride_qd),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_dq = dQ_ptr + b_index * stride_qb + h_index * stride_qh
    dQ_block_ptr = tl.make_block_ptr(
        base_dq,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qs, stride_qd),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_k = K_ptr + b_index * stride_kb + (h_index // GROUP_SIZE) * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base_k,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_ks, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    
    base_v = V_ptr + b_index * stride_vb + (h_index // GROUP_SIZE)  * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base_v,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    
    base_o = O_ptr + b_index * stride_ob + h_index * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base_o,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_os, stride_od),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_do = dO_ptr + b_index * stride_ob + h_index * stride_oh
    dO_block_ptr = tl.make_block_ptr(
        base_do,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_os, stride_od),
        offsets=(q_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_MODEL),
        order=(1, 0),
    )
    base_d = D_ptr + b_index * stride_lb + h_index * stride_lh
    D_block_ptr = tl.make_block_ptr(
        base_d,
        shape=(N_QUERIES,),
        strides=(stride_ls, ),
        offsets=(q_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )
    base_l = L_ptr + b_index * stride_lb + h_index * stride_lh
    L_block_ptr = tl.make_block_ptr(
        base_l,
        shape=(N_QUERIES,),
        strides=(stride_ls, ),
        offsets=(q_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,),
    )

    # Pointers for K, V gradients (cannot use block pointers for atomic add)
    base_dk = dK_ptr + b_index * stride_kb + (h_index // GROUP_SIZE)  * stride_kh
    base_dv = dV_ptr + b_index * stride_vb + (h_index // GROUP_SIZE)  * stride_vh

    k_offsets = tl.arange(0, K_TILE_SIZE)   # (BK, 1) k_start + 
    d_offsets = tl.arange(0, D_MODEL)                      # (1, D_MODEL)

    dK_tile_ptrs = base_dk + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd   # (BK, D_MODEL)
    dV_tile_ptrs = base_dv + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd

    # Initialize buffers for dQ (NOTE: what about dtype?)
    dQ = tl.zeros((Q_TILE_SIZE, D_MODEL), dtype=tl.float32) # (Q_TILE_SIZE, D_MODEL)
    
    # Load the current tile of Q, O, dO, and L
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D_MODEL)
    O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D_MODEL)
    dO = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_TILE_SIZE, D_MODEL)
    D = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero") # (Q_TILE_SIZE,)

    q_start = q_index * Q_TILE_SIZE
    for k_iter in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current tiles of K, V
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D_MODEL)
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D_MODEL)

        # Re-Compute scaled tile score
        S = tl.dot(Q, tl.trans(K)) * scale # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal mask
        if is_causal:
            k_start = k_iter * K_TILE_SIZE
            q_ids = q_start + tl.arange(0, Q_TILE_SIZE)
            k_ids = k_start + k_offsets
            mask = (q_ids[:, None] >= k_ids[None, :]) & (k_ids[None, :] < N_KEYS) & (q_ids[:, None] < N_QUERIES)
            S = tl.where(mask, S, S - 1e6)

        # Compute exponents of scores
        P = tl.exp(S.to(tl.float32) - L[:, None])

        # Compute dV
        dV = tl.dot(tl.trans(P), dO.to(tl.float32))

        # Compute dP and dS
        # dP = tl.dot(dO, tl.trans(V))
        dP = tl.dot(dO.to(tl.float32), tl.trans(V).to(tl.float32))

        dS = P * (dP - D[:, None]) * scale

        # Compute dQ, dK
        # dQ += tl.dot(dS, K)
        dQ += tl.dot(dS, K.to(tl.float32))
        # dK = tl.dot(tl.trans(dS), Q)
        dK = tl.dot(tl.trans(dS), Q.to(tl.float32))

        # Atomically add output of the dK, dV to global memory 
        mask_kd = (k_offsets[:, None] + k_iter * K_TILE_SIZE< N_KEYS) # - k_iter * K_TILE_SIZE)  # (BK,1) broadcastable to (BK, D_MODEL)
        # tl.atomic_add(dK_tile_ptrs, dK.to(tl.float32), mask=mask_kd)
        tl.atomic_add(dK_tile_ptrs, dK.to(tl.bfloat16), mask=mask_kd)
        # tl.atomic_add(dV_tile_ptrs, dV.to(tl.float32), mask=mask_kd)
        tl.atomic_add(dV_tile_ptrs, dV.to(tl.bfloat16), mask=mask_kd)

        # Move the pointers to the next tile.
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0)) # (K_TILE_SIZE, D_MODEL)
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0)) # (K_TILE_SIZE, D_MODEL)
        dK_tile_ptrs += K_TILE_SIZE * stride_ks
        dV_tile_ptrs += K_TILE_SIZE * stride_vs
        
    # Write output to the dQ V block pointers
    tl.store(dQ_block_ptr, dQ.to(Q.dtype), boundary_check=(0,1)) # (Q_TILE_SIZE, D_MODEL)
