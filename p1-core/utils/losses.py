__all__ = [
    "softmax",
    "cross_entropy",
    "perplexity",
    "get_loss_fn",
]

import torch
from einops import rearrange

def softmax(x: torch.Tensor, dim: int, tau: float = 1.0) -> torch.Tensor:
    assert -x.dim() <= dim < x.dim(), "Dimension is wrong"
    assert tau > 0, "Temperature must be positive."
    # force stability
    if torch.is_autocast_enabled():
        x = x.float()
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp((x - x_max) / tau)
    return exps / torch.sum(exps, dim = dim, keepdim=True)

def cross_entropy(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = 0.0) -> torch.Tensor:
    assert logits.shape[:-1] == target.shape, "logits and target shape dimension mismatch"
    # Force stability: keep logits in fp32 under AMP
    if torch.is_autocast_enabled():
        logits = logits.float()
    # Flatten all dimensions except vocab
    logits_flat = rearrange(logits, "... v_size -> (...) v_size")
    target_flat = rearrange(target, "... -> (...)")

    # Subtract max logit per row
    m = logits_flat.max(dim=-1, keepdim=True).values
    logits_adj = logits_flat - m

    # Save stabilized logits
    B = logits_adj.shape[0]
    logits_idx = logits_adj[torch.arange(B, device=target.device), target_flat]
    log_z = torch.log(torch.sum(torch.exp(logits_adj), dim = -1))

    # Calculate loss: Negative log-likelihood and log_z
    losses = -logits_idx + log_z
    if z_alpha > 0:
        losses += z_alpha * log_z**2
    return torch.mean(losses)

def perplexity(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy(logits, target, 0.0))

def get_loss_fn(loss_cfg: dict):
    loss_name = loss_cfg.get("name", "cross_entropy")
    if loss_name != "cross_entropy":
        raise NotImplementedError(f"Only 'cross_entropy' is supported, but '{loss_name}' was provided.")
    
    z_alpha = loss_cfg.get("z_alpha", 0.0)
    def loss_fn(logits: torch.Tensor, target: torch.Tensor):
        return cross_entropy(logits, target, float(z_alpha))

    return loss_fn