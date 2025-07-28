import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import os
from time import time, perf_counter
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

def train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params):
    # unpack params
    steps = run_params["steps"]
    batch_size = run_params["batch_size"]
    context_length = run_params["context_length"]
    valid_freq = run_params["valid_freq"]
    valid_total = run_params["valid_total"]
    serialize_path = run_params["serialize_path"]
    serialize_first = run_params["serialize_first"]
    serialize_freq = run_params["serialize_freq"]    
    run_name = run_params["run_name"]
    device = run_params["device"]
    lr = scheduler_params["lr_max"]

    # init logging and serialization
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    ckpt_path = Path(serialize_path) / run_name / "ckpt_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.train()
    loop = tqdm(range(steps), leave = True)
    valid_loss, min_train_loss, min_valid_loss = float('nan'), float('inf'), float('inf')
    for step in loop:
        # update learning rate TODO: create one-line function
        scheduler_params["t"] = step + 1
        lr = cosine_lr_schedule(**scheduler_params)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
           
        # chose ids of words
        tokens_curr, tokens_next = data_loading(tokens["train"], batch_size, None, context_length, device)
        # calculate training logits and loss
        logits = model(tokens_curr)
        loss = loss_fn(logits, tokens_next)
        if loss.item() < min_train_loss:
            min_train_loss = loss.item()
        # log train params to tensorboard
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/perplexity", np.exp(loss.item()), step)

        # log validate params and log in tensorboard
        if step % valid_freq == 0:
            valid_loss = get_valid_loss(tokens["valid"], model, loss_fn, context_length, batch_size, None, device)
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                if step / steps > serialize_first and (step % serialize_freq == 0):
                    save_checkpoint(model, optimizer, step, ckpt_path)

            # add validate scalars to Tensorboard
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/perplexity", np.exp(valid_loss), step)


        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()

        # clip gradients TODO: add to parser
        if max_norm is not None:
            gradient_clipping(model.parameters(), max_l2_norm= max_norm)
        
        optimizer.step()

        # update progress bar
        loop.set_postfix(lr = lr, train_loss=loss.item(), min_train_loss=min_train_loss, valid_loss=valid_loss, min_valid_loss=min_valid_loss)

    # run the best model on the full valid set
    if not ckpt_path.exists():
        save_checkpoint(model, optimizer, -1, ckpt_path)
    n_iter = load_checkpoint(ckpt_path, model, optimizer)
    t = perf_counter()
    final_valid_loss = get_valid_loss(tokens["valid"], model, loss_fn, context_length, batch_size, valid_total, device)
    final_perplexity = np.exp(final_valid_loss)
    writer.add_text(
        "summary/final_valid_metrics",
        f"Full valid loss: {final_valid_loss:.4f}, Full valid perplexity: {final_perplexity:.2f}, Number of samples: {valid_total:,}, Time={perf_counter() - t:.2f}s",
        step+1
    )
    print(f"⏱️ Full validation for {valid_total} samples is {final_valid_loss:.4f}. Time={perf_counter() - t:.2f}s") 
    
    # close writer
    writer.close()

def main(config):
    model_params, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    max_norm = clip_grad_params["max_norm"]

    # get experiments
    model = TransformerLM(**model_params)
    optimizer_params["params"] = model.parameters()
    optimizer = Adam(**optimizer_params)
    tokens = {
        "train": np.load(tokens_params["train"], mmap_mode='r'),
        "valid": np.load(tokens_params["valid"], mmap_mode='r')
    }
    loss_fn = cross_entropy # TODO: probably, it makes sense to automate
    
    # run training loop
    train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params)


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
