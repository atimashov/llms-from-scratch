__all__ = [
    "sample_indices",
    "get_start_seqs",
    "data_loading",
]

import numpy as np
import numpy.typing as npt
import torch
from time import perf_counter

def sample_indices(loader_mode: str, train_id_range: int, valid_id_range: int, num_train_ckpt_report: tuple[int, int, int]):
    """
    Sample training and validation token indices.

    Notes:
    - Validation indices should be fixed for fair comparison.
    - Deterministic first 1280/32K tokens give unusually bad validation results.

    Args:
        loader_mode: Sampling strategy ("sample" or "in_memory_ids")
        train_id_range: Total range of train indices to sample from
        valid_id_range: Total range of validation indices to sample from
        num_train_ckpt_report: Tuple of (num_train, num_ckpt_valid, num_report_valid) - how many indices to sample.

    Returns:
        im_ids_train: np.ndarray | None
        im_ids_valid: dict of validation index arrays
    """
    num_train, num_ckpt, num_report = num_train_ckpt_report 
    
    # Training sampling
    im_ids_train = None
    if loader_mode == "in_memory_ids":
        t0 = perf_counter()
        im_ids_train = np.random.choice(train_id_range, size=min(num_train, train_id_range), replace=False)
        print(f"Sampled to keep in-memory train ids: {im_ids_train.shape[0]:,} training positions. Time={perf_counter() - t0:.2f}s") 

    # Validation sampling
    t1 = perf_counter()
    train_sample_source = train_id_range if im_ids_train is None else im_ids_train
    source_train_len = train_sample_source if isinstance(train_sample_source, int) else train_sample_source.shape[0]
    im_ids_valid = {
        "train_report": np.random.choice(train_sample_source, size=min(num_report, source_train_len), replace=False),
        "valid_ckpt": np.random.choice(valid_id_range, size=min(num_ckpt, valid_id_range), replace=False),
        "valid_report": np.random.choice(valid_id_range, size=min(num_report, valid_id_range), replace=False)
    }
    print(
        f"Sampled validation indices: "
        f"train report={im_ids_valid['train_report'].shape[0]:,}, "
        f"ckpt={im_ids_valid['valid_ckpt'].shape[0]:,}, "
        f"report={im_ids_valid['valid_report'].shape[0]:,}. "
        f"Time={perf_counter() - t1:.2f}s"
    )
    return im_ids_train, im_ids_valid

def get_start_seqs(start_from: int | None, bs: int | None, x_len: int | None, in_memory_ids: np.ndarray | None, mode : str):
    assert mode in {"sample", "in_memory_ids"}
    if mode == "sample":
        start_seqs = np.random.randint(0, x_len, size=bs)[:, None]
    elif mode == "in_memory_ids":
        start_from = start_from % in_memory_ids.shape[0] 
        start_seqs = in_memory_ids[start_from:start_from + bs]
        if start_from + bs > in_memory_ids.shape[0]:
            offset = start_from + bs - in_memory_ids.shape[0]
            start_seqs = np.concat((start_seqs, in_memory_ids[:offset]), axis = 0)
        start_seqs = start_seqs[:, None]
    return start_seqs

def data_loading(x: npt.NDArray, context_length: int, start_seqs: np.ndarray, device: torch.device | None = None) -> (torch.Tensor, torch.Tensor):
    """
    Create batch of data to train.

    Args:

    Returns:
        tokens_curr:
        tokens_next:

    """
    # create masks to sample from numpy
    steps_curr = np.arange(context_length)[None, :]
    steps_next = np.arange(1, context_length + 1)[None, :]
    mask_curr, mask_next = start_seqs + steps_curr, start_seqs + steps_next
    # sample numpy tokens
    tokens_curr_np = x[mask_curr]
    tokens_next_np = x[mask_next]
    # convert to PyTorch (NOTE: how dtype conversion slows this down?)
    tokens_curr = torch.from_numpy(tokens_curr_np).to(device = device, dtype = torch.int)
    tokens_next = torch.from_numpy(tokens_next_np).to(device = device, dtype = torch.int)
    return tokens_curr, tokens_next