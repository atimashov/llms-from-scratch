__all__ = [
    "count_parameters",
    "get_model_memory",
    "get_activations_memory",
    "get_expected_memory",
    "print_memory_stats",
    "est_forward_flops",
    "print_d_model_d_ff",
]

from termcolor import colored

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_memory(config):
    """
    Calculate memory for model parameters (no bias, activations handled separately).
    If AMP is used, weight keep the same dtype, only activations change.

    Memory estimate:
        - float16/bf16: ~ 2 x [2 x V x d + 12 x L x d^2]
        - FP32              : ~ 4 x [2 x V x d + 12 x L x d^2]
    Returns:
        total_param_bytes, rope_buffer_bytes
    """
    # extract variables
    BYTES_IN_MB = 1024 ** 2
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    seq_len = config["model"]["context_length"]
    ff_ratio = 3 if config["model"] else 2
    n_layers = config["model"]["num_layers"]
    n_heads = config["model"]["num_heads"]
    vocab_size = config["model"]["vocab_size"]
    norms = config["model"]["norms"]
    num_emb_matrices = 1 if config["model"]["weights_tying"] else 2
    dtype_bytes = 4 if config["model"]["dtype"] in {"float", "float32", "amp"} else 2
    
    # Embedding parameters and Final linear layer
    numel_emb = vocab_size * d_model * num_emb_matrices # NOTE: ~ V x d OR 2 x V x d

    # Norm size estimates
    rmsnorm, lnorm = d_model, 2 * d_model

    # Norms inside each transformer block
    numel_norms_tb = 0
    if norms["before"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_tb += (rmsnorm if norms["before"] == "RMSNorm" else lnorm)  # (attn & ffn)
    if norms["after"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_tb += (rmsnorm if norms["after"] == "RMSNorm" else lnorm)  # (attn & ffn)

    # Final norm
    numel_norms_f = 0
    if norms["final"] in {"RMSNorm", "LayerNorm"}:
        numel_norms_f += (rmsnorm if norms["final"] == "RMSNorm" else lnorm)

    # Transformer block parameters (per layer)
    numel_tb = 4 * d_model ** 2 + numel_norms_tb # P, Q, V, O
    numel_tb += ff_ratio * d_ff * d_model + numel_norms_tb # ~ 4d^2 + numel_norms_tb

    # Final Memory
    numel = numel_emb + numel_norms_f + n_layers * numel_tb
    memory = dtype_bytes * numel

    # RoPE buffer
    buffer = dtype_bytes * seq_len * d_model / n_heads 

    # Mask buffer
    buffer += seq_len ** 2
    
    return memory / BYTES_IN_MB, buffer / BYTES_IN_MB

def get_activations_memory(config):
    """
    Estimate expected memory usage (in MB) for activations.
    Input counted once (act_emb). Per layer: attention + FFN activations.

    Memory estimate:
        - AMP (float16/bf16): ~ 2 B x S x [3d + L x (18d + 2H x S) + 2V - 2HS]
        - FP32              : ~ 4 B x S x [2d + L x (16d + 2H x S) + V - HS]
    Returns:
        total_activation_MB
    """
    BYTES_IN_MB = 1024 ** 2
    # extract variables
    B = config["train"]["batch_size"]
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    S = config["model"]["context_length"]
    V = config["model"]["vocab_size"]
    L = config["model"]["num_layers"]
    H = config["model"]["num_heads"]
    gated = config["model"]["is_gate"]
    norms = config["model"]["norms"]

    # memory per item
    dtype = config["model"]["dtype"]
    is_fp32 = dtype in {"float", "float32"}
    is_fp16 = dtype == "float16"
    is_amp  = dtype == "amp"
    bytes_model = 4 if (is_fp32) else 2
    bytes_up = 4 if (is_fp32 or is_amp) else 2

    # sizes reuse
    BSd = B * S * d_model
    BSf = B * S * d_ff
    BHS2 = B * H * S * S
    BSV = B * S * V
    
    # norms present
    n_before = int(norms["before"] in {"RMSNorm", "LayerNorm"})
    n_after = int(norms["after"] in {"RMSNorm", "LayerNorm"})
    n_final = int(norms["final"] in {"RMSNorm", "LayerNorm"})

    # 0. Embeddings activations (saved once)
    mem_emb = bytes_model * BSd
    
    # 1. Attention activations (per layer)
    act_qkv = 3 * BSd # no RoPE 
    act_attn = BSd
    act_proj = BSd
    act_wght = BHS2
    act_sm = 0 if is_amp else BHS2 # for some reason for AMP it is not counted

    mem_mha_model = bytes_model * L * (act_qkv + act_attn + act_proj)
    mem_mha_up = bytes_up * (L * (act_wght + act_sm) - act_wght)
    mem_mha_norm = bytes_up * L * (n_before + n_after) * BSd
    mem_mha = mem_mha_model + mem_mha_up + mem_mha_norm

    # 2. FF block activations (per layer)
    hidden_mult = 3 if gated else 2

    mem_ff_model = bytes_model * L * (hidden_mult * BSf + BSd)
    mem_ff_norm = bytes_up * L * (n_before + n_after) * BSd
    mem_ff = mem_ff_model + mem_ff_norm

    # 3. Final layer activations (looks like only logits saved in max dtype)
    mem_final = bytes_up * (BSV + BSd * n_final)

    # 4 Memory total
    mem = mem_emb + mem_mha + mem_ff + mem_final

    return mem / BYTES_IN_MB


def get_expected_memory(config):
    """
    Estimate expected memory usage (in MB) for training.
    
    Breakdown:
        I.   Model parameters
        II.  Gradients
        III. Optimizer state
        IV.  Input tokens
        V.   Activations (for backward)
    """
    # extract variables
    BYTES_IN_MB = 1024 ** 2
    bs = config["train"]["batch_size"]
    seq_len = config["model"]["context_length"]
    vocab_size = config["model"]["vocab_size"]
    dtype_bytes_input = 2 if vocab_size <= 65536 else 4
    
    # --------- I. model ---------
    memory_model, buffer_model = get_model_memory(config)
    
    # ------- II. gradients ------
    memory_grad = memory_model
    
    # --- III. optimizer state ---
    # supported 'Lion', 'Adam', 'AdamW', 'Adan'
    memory_os = memory_grad if config["optimizer"]["name"] in {'Lion'} else 2 * memory_grad 
    
    # --------- IV. Input --------
    memory_input = dtype_bytes_input * bs * seq_len / BYTES_IN_MB

    # ------ V. Activations ------
    memory_act = get_activations_memory(config)

    memory_steady = memory_model + buffer_model + memory_grad + memory_os + memory_input
    memory_peak = memory_steady + memory_act
    return {
        "steady_MB": memory_steady,
        "peak_MB": memory_peak,
        "model_MB": memory_model,
        "buffer_MB": buffer_model,
        "grad_MB": memory_grad,
        "optim_MB": memory_os,
        "input_MB": memory_input,
        "activations_MB": memory_act,
    }

def print_memory_stats(memories):
    print(
        f"{colored("Memory consumption, MB: ", 'blue')}"
        f"{colored('Peak=', 'blue')}{memories['peak_MB']:,.2f} | "
        f"{colored('Steady=', 'blue')}{memories['steady_MB']:,.2f} | "
        f"{colored('Inputs=', 'blue')}{memories['input_MB']:,.2f} | "        
        f"{colored('Model=', 'blue')}{memories['model_MB']:,.2f} | "
        f"{colored('Buffer=', 'blue')}{memories['buffer_MB']:,.2f} | "
        f"{colored('Grads=', 'blue')}{memories['grad_MB']:,.2f} | "
        f"{colored('Opt states=', 'blue')}{memories['optim_MB']:,.2f} | "
        f"{colored('Activations=', 'blue')}{memories['activations_MB']:,.2f} | "
    )

def est_forward_flops(config):
    """
    FLOPs estimate (forward pass):
    - Attention per layer: ~8BSD^2 + 4BS^2D
    - FFN per layer: (4 if non-gated else 6) BSDD_ff
    - Final projection: 2 BSDV

    If d_ff ~ 4d (non-gated) or 8/3d (gated), TOTAL FLOPS:
    - ~ 2BSD x [L x (12D + 2S)  + V]
    - (per token) ~ 2D x [L x (12D + 2S)  + V]
    """
    # extract variables
    B = config["train"]["batch_size"]
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    S = config["model"]["context_length"]
    V = config["model"]["vocab_size"]
    L = config["model"]["num_layers"]
    gated = config["model"]["is_gate"]

    # Transformer block FLOPS (per layer)
    flops_attn = 4 * (2 * B * S * d_model ** 2) + 2 * (2 * B * S ** 2  * d_model)
    hidden_mult = 3 if gated else 2
    flops_ff = hidden_mult * 2 * B * S * d_model * d_ff # in both cases ~16BSD^2

    # Final Projection
    flops_proj = 2 * B * S * d_model * V
    
    return (L * (flops_attn + flops_ff) + flops_proj) // (B * S)

def print_d_model_d_ff(d_model: int, d_ff: int, is_gate: bool):
    ratio = 8/3 if is_gate else 4
    deviation = 100 * (max(d_ff / d_model, ratio) / min(d_ff / d_model, ratio) - 1)
    dev_print = colored(f"Deviaton is {deviation:.1f}%", "red" if deviation > 5 else  "green")
    draft_print = colored("Activation ", "blue") + f"is {'' if is_gate else 'not '}gated - d_ff/d_model is expected ~{ratio:.2f}; {dev_print}. "
    rec = 'good' if d_ff % 64 == 0 else 'not recommended'
    div64_print = f"Hidden dim is {'' if d_ff % 64 == 0 else 'not '}divisible by 64. It is {rec} for GPU."
    print(draft_print + div64_print)