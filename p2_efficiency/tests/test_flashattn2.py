import pytest
import torch

from p1_core.layers.attention import scaled_dot_product_attention as naive_attn
from p2_efficiency.kernels.flashattn2 import FlashAttention2
from p2_efficiency.profiling import generate_random_qkv

def _run_attn(attn_fn, Q, K, V, is_causal = True, do_bwd = True):
    out = attn_fn(Q = Q, K = K, V = V, is_causal = is_causal)
    loss = out.sum()
    loss.backward()
    grads = (Q.grad.detach(), K.grad.detach(), V.grad.detach())
    
    return out.detach(), grads

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 4, 64, 32), (4, 8, 128, 16)]) # (B, H, S, D)
@pytest.mark.parametrize("is_causal", [True, False])
def test_flashattn2_matches_naive(dtype, shape, is_causal):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    B, H, S, D = shape
    device = "cuda:0"
    atol = 2e-2 if dtype == torch.float16 else 3e-2
    rtol = 2e-2 if dtype == torch.float16 else 3e-2

    Q, K, V = generate_random_qkv(B, H, H, S, D * H, device, dtype, True)

    # 1. Run Naive Attention
    out_ref, grad_ref = _run_attn(naive_attn, Q, K, V, is_causal)

    # 2. Run Flash Attention
    def flash(Q, K, V, is_causal):
        return FlashAttention2.apply(Q, K, V, is_causal, 64, 64, 4, 2)
    Q.grad, K.grad, V.grad = None, None, None
    out_fa2, grad_fa2 = _run_attn(flash, Q, K, V, is_causal)

    # 3. Compare outputs and gradients
    assert torch.allclose(out_ref, out_fa2, atol=atol, rtol=rtol)

    gQ_ref, gK_ref, gV_ref = grad_ref
    gQ_fa2, gK_fa2, gV_fa2 = grad_fa2
    assert torch.allclose(gQ_ref, gQ_fa2, atol=atol, rtol=rtol)
    assert torch.allclose(gK_ref, gK_fa2, atol=atol, rtol=rtol)
    assert torch.allclose(gV_ref, gV_fa2, atol=atol, rtol=rtol)