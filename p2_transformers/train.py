import torch
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import os
from time import time
from datetime import datetime
import numpy as np
import platform

from models import *
from utils import *
from pos_enc import *
from norms import *
from optim import *
from bpe_tokenizer import *



seed = 123
torch.manual_seed(seed)

def train_loop(steps, model, optimizer, scheduler_params, loss_fn, tokens_data, batch_size, context_length, device, dtype, lr):
    model.train()  # put model to training mode TODO: should I do it every step?
    loop = tqdm(range(steps), leave = True)
    for n_step in loop:
        # update learning rate
        if scheduler_params is not None:
            scheduler_params["t"] = n_step + 1
            lr = cosine_lr_schedule(**scheduler_params)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
           
        # chose ids of words
        tokens_curr, tokens_next = data_loading(tokens_data, batch_size, context_length, device)
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)

        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()

        # clip gradients TODO: add condition
        gradient_clipping(model.parameters(), max_l2_norm= 1.0)
        optimizer.step()
        
        # TODO: serialize
        # TODO: do logging
        
        # update progress bar
        loop.set_postfix(lr = lr, loss=loss.item())


def main(config):
    # train
    batch_size = config["train"]["batch_size"]
    context_length = config["model"]["context_length"]
    steps = int(config["train"]["total_tokens_processed"] / (batch_size * context_length))

    # model
    if config["gpu"] is None:
        device = "mps" if platform.system() == "Darwin" else "cpu"
    else:
        device = "cuda:{}".format(config["gpu"])
    model_params = {
        "d_model": config["model"]["d_model"],
        "d_ff": config["model"]["d_ff"],
        "num_heads": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "theta": config["model"]["rope_theta"],
        "context_length": config["model"]["context_length"],
        "vocab_size": config["model"]["vocab_size"],
        "device": torch.device(device),
        "dtype": torch.float32 # TODO: improve later
    }
    model = TransformerLM(**model_params)
    # optimizer
    optim_params = {
        "params": model.parameters(),
        "lr": float(config["optimizer"]["lr"]),
        "betas": (config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        "weight_decay": config["optimizer"]["weight_decay"],
        "eps": float(config["optimizer"]["epsilon"]),
        "decoupled": True,
    }
    optimizer = Adam(**optim_params)
    if "scheduler" in config["optimizer"]:
        sched_params = {
            "t": 1,
            "lr_max": float(config["optimizer"]["lr"]),
            "lr_min": float(config["optimizer"]["scheduler"]["lr_min"]),
            "warmup_iters": int(config["optimizer"]["scheduler"]["warmup_iters"] * steps),
            "cosine_cycle_iters": (config["optimizer"]["scheduler"]["cosine_cycle_iters"] * steps),
        }
    else: 
        sched_params = None
    
    # loss 
    loss_fn = cross_entropy

    # data: tokenize
    if config["dataset_path"]["tokenized"] == "None":
        bpe_params = {
            "input_path": config["dataset_path"]["raw"], 
            "vocab_size": config["model"]["vocab_size"], 
            "special_tokens": ["<|endoftext|>"]
        }
        bpe = BPETokenizer(**bpe_params)
        # train BPE tokenizer
        bpe.train()
        # encode text (TODO: test "encode_iterable")
        print(datetime.now())
        with open(bpe_params["input_path"], 'r') as f:
            text = f.read()
        token_ids = bpe.encode(text, num_processes= 24)
        # TODO :save it to use as np.memmap
        print(datetime.now())
        token_ids_np = np.array(token_ids, dtype=np.int32)
        print(datetime.now())
        ext = config["dataset_path"]["raw"].split(".")[-1]
        config["dataset_path"]["tokenized"] = config["dataset_path"]["raw"].replace(f".{ext}", ".npy")
        np.save(config["dataset_path"]["tokenized"], token_ids_np)
        print(datetime.now())
    # data: get tokenized data
    tokens_data = np.load(config["dataset_path"]["tokenized"], mmap_mode='r') 

    train_loop(steps, model, optimizer, sched_params, loss_fn, tokens_data, batch_size, context_length, model_params["device"], model_params["dtype"], optim_params["lr"])



if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    inputs = parser.parse_args()
    print(inputs)

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    main(config)
