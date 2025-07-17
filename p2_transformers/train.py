import torch
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import os
from time import time
import numpy as np
import platform

from models import *
from utils import *
from pos_enc import *
from norms import *
from bpe_tokenizer import *



seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


def train_loop(steps, model, optimizer, loss_fn, tokens_data, batch_size, context_length, device, dtype):
    model.train()  # put model to training mode TODO: should I do it every step?
    loop = tqdm(range(steps), leave = True)
    for n_step in loop:
        # TODO: update optmizer
        tokens_curr, tokens_next = data_loading(tokens_data, batch_size, context_length, device)
        logits = model(token_curr)
        loss = loss_fn(logits, tokens_next)

        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()
        optimizer.step()
        
        # TODO: serialize
        # TODO: do logging
        
        # update progress bar
        loop.set_postfix(loss=loss.item())




def main(config):
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
    opt_params = {
        "params": model.parameters(),
        "lr": config["optimizer"]["lr"],
        "betas": (config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        "weight_decay": config["weight_decay"],
        "eps": config["epsilon"],
        "decoupled": True
    }
    optimizer = Adam(**opt_params)
    # scheduler:
    #     name: cosine
    #     lr_min: 1e-6
    #     warmup_iters: 0.1
    #     cosine_cycle_iters: 1.0
    
    # loss 
    loss_fn = cross_entropy

    # data: tokenize
    if config["dataset_path"]["tokenized"] is None:
        bpe_params = {
            "input_path": config["dataset_path"]["raw"], 
            "vocab_size": config["model"]["vocab_size"], 
            "special_tokens": ["<|endoftext|>"]
        }
        bpe = BPETokenizer(**bpe_params)
        # train BPE tokenizer
        bpe.train()
        # encode text (TODO: test "encode_iterable")
        with open(bpe_params["input_path"], 'r') as f:
            text = f.read()
        token_ids = bpe.encode(text, num_processes= 24)
        # TODO :save it to use as np.memmap
        token_ids_np = np.array(token_ids, dtype=np.int32)
        ext = config["dataset_path"]["raw"].split(".")[-1]
        config["dataset_path"]["tokenized"] = config["dataset_path"]["raw"].replace(f".{ext}", ".npy")
        np.save(config["dataset_path"]["tokenized"], token_ids_np)
    # data: get tokenized data
    tokens_data = np.load(config["dataset_path"]["tokenized"], mmap_mode='r') 
    
    # train
    batch_size = config["train"]["batch_size"]
    context_length = config["model"]["context_length"]
    steps = int(config["train"]["total_tokens_processed"] / (batch_size * context_length))
    train_loop(steps, model, optimizer, loss_fn, tokens_data, batch_size, context_length, model_params["device"], model_params["dtype"])



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
