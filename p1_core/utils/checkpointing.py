__all__ = [
    "save_checkpoint",
    "load_checkpoint",
]

import torch
import yaml

def save_checkpoint(model, optimizer, scaler, scheduler_params: dict, iteration, loss_checkpoint, loss_report, out_path, config: dict | None = None):
    model_cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    # Build checkpoint object
    obj = {
        "model": model_cpu_state,
        "optimizer": optimizer.state_dict(),
        "iter_number": iteration,
        "loss_checkpoint": loss_checkpoint,
        "loss_report": loss_report,
    }
    if scaler is not None:
        obj["scaler"] = scaler.state_dict()
    if isinstance(scheduler_params, dict):
        obj["scheduler_params"] = scheduler_params
    
    # Save checkpoint
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, out_path)
    
    # Save config if provided
    if config is not None:
        config_path = out_path.parent / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

def load_checkpoint(src, model, optimizer = None, scaler = None, device = "cpu"):
    obj = torch.load(src, map_location = device)
    
    # Load model weights
    model.load_state_dict(obj["model"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])

    # Load AMP scaler state if provided
    if scaler is not None and "scaler" in obj:
        scaler.load_state_dict(obj["scaler"])

    # Load scheduler config if available
    scheduler_cfg = obj.get("scheduler_params", None)
    
    return (
        obj["iter_number"],
        obj.get("loss_checkpoint", None),
        obj.get("loss_report", None),
        scheduler_cfg
    )