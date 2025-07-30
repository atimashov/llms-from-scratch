import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import yaml
from argparse import ArgumentParser
from pathlib import Path
import os
from time import perf_counter
from datetime import datetime
import numpy as np
import platform

from models import TransformerLM
from utils import parse_config, cross_entropy, cosine_lr_schedule, data_loading, get_valid_loss, gradient_clipping, save_checkpoint, load_checkpoint
from optimizers import Adam


seed = 123
torch.manual_seed(seed)

class Generator:
    """
    TODO:
        - memory preallocation for prompt
        - KV Cache
        - top-p sampling
        - AMP/FP16
        - batched generation
        - parallel decoding
    """
    def __init__(self, tokenizer, model, max_num_tokens: int | None = None, device: torch.device = torch.device("cpu")):
        # TODO: implement KV cache
        self.device = device
        self.tokenizer = tokenizer
        self.eof_token = self.tokenizer.eof_token
        self.model = model
        self.n_steps = float('inf') if max_num_tokens is None else max_num_tokens

    def generate_next(self, tokens, tau: float = 1.0, topk: int | None = None):
        pred = self.model(tokens, prob = True, tau = tau)[0, -1] # 
        # sample
        if topk is None:
            topk = pred.shape[0]
        values, indices = torch.topk(pred, topk)
        probs = value / values.sum()
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return indices[sampled_idx].item()

    def generate_tokens(self, tokens, tau: float, topk: int | None = None):
        curr_pred = None
        step = 0 
        tokens_pred = []
        while curr_pred != self.eof_token and step < self.n_steps:
            curr_pred = self.generate_next(tokens, tau, topk)
            add = torch.tensor([[curr_pred]]).to(device = self.device, dtype = torch.long)
            tokens = torch.cat((tokens, add), dim = 1) # TODO: probably can do more efficiently, for example pre-allocate memory
            step += 1
            if curr_pred != self.eof_token:
                tokens_pred.append(curr_pred)
        return tokens_pred

    def generate(self, prompt: str, tau: float, topk: int | None = None):
        # encode
        tokens_list = self.tokenizer.encode(prompt)
        tokens = torch.as_tensor(tokens_list, device = self.device, dtype = torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        # generate
        tokens_pred = self.generate_tokens(tokens, tau, topk)
        # decode TODO: modify to decode on the fly
        return self.tokenizer.decode(tokens_pred)

            

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    # parser.add_argument('--log-structure', type=str, default='rtx4000ada', help = 'subfolders structure inside run to log')
    inputs = parser.parse_args()
    print(inputs)

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    for lr in [5e-3, 1e-3, 1e-4]: # 1e+1 # 1e-0, 1e-1, 5e-2, 
        config["optimizer"]["lr"] = lr
        for ratio in [1e-0, 1e-1, 1e-2, 1e-3]:
            config["optimizer"]["scheduler"]["lr_min"] = round(lr * ratio, 7)
            main(config)
    
    # if config["gpu"] == 0:
    #     for lr in [10.0, 1.0, 1e-1, 1e-2, 5e-3, 1e-3, 1e-4]:
    #         config["optimizer"]["lr"] = lr
    #         for ratio in [1.0, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]:
    #             config["optimizer"]["scheduler"]["lr_min"] = round(lr * ratio, 6)
    #             main(config, inputs.log_structure)
    # elif config["gpu"] == 1:
    #     for lr in [10.0, 1.0, 1e-1, 1e-2]:
    #         config["optimizer"]["lr"] = lr
    #         for ratio in [1.0, 1e-1, 1e-2]:
    #             config["optimizer"]["scheduler"]["lr_min"] = round(lr * ratio, 6)
    #             main(config, inputs.log_structure)
    # elif config["gpu"] == 1:
    #     for lr in [7e-3, 4e-3, 1e-3]: # [5e-2, 3e-2, 1e-2, 5e-3]:
    #         config["optimizer"]["lr"] = lr
    #         for ratio in [2e-1, 1e-1, 5e-2]: #[1e-1, 5e-2]:
    #             config["optimizer"]["scheduler"]["lr_min"] = round(lr * ratio, 6)
    #             main(config, inputs.log_structure)
