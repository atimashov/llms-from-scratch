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

def train_loop(steps, model, optimizer, scheduler_params, loss_fn, tokens_data, batch_size, context_length, device, dtype, lr, out_path, run_name: str, freq_validate: int = 10, not_serialize: float = 0.3):
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    ckpt_path = Path(out_path) / run_name / "ckpt_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)  # creates all parent dirs
    
    model.train()
    loop = tqdm(range(steps), leave = True)
    val_loss, min_train_loss, min_val_loss = float('nan'), float('inf'), float('inf')
    for step in loop:
        # update learning rate
        if scheduler_params is not None:
            scheduler_params["t"] = step + 1
            lr = cosine_lr_schedule(**scheduler_params)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
           
        # chose ids of words
        tokens_curr, tokens_next = data_loading(tokens_data["train"], batch_size, None, context_length, device)
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
        if step % freq_validate == 0:
            val_loss = get_val_loss(tokens_data["validate"], model, loss_fn, context_length, batch_size, None, device)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                if (step + 1) / steps > not_serialize:
                    save_checkpoint(model, optimizer, step, ckpt_path)

            # add validate scalars to Tensorboard
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/perplexity", np.exp(val_loss), step)


        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()

        # clip gradients
        if "clip_gradient" in config["optimizer"]:
            max_norm = config["optimizer"]["clip_gradient"].get("max_norm", None)
            if max_norm is not None:
                gradient_clipping(model.parameters(), max_l2_norm= max_norm)
        
        optimizer.step()

        # update progress bar
        loop.set_postfix(lr = lr, train_loss=loss.item(), min_train_loss=min_train_loss, val_loss=val_loss, min_val_loss=min_val_loss)

    # run the best model on the full val set
    if not ckpt_path.exists():
        save_checkpoint(model, optimizer, -1, ckpt_path)
    n_iter = load_checkpoint(ckpt_path, model, optimizer)
    t = perf_counter()
    final_val_loss = get_val_loss(tokens_data["validate"], model, loss_fn, context_length, batch_size, batch_size * 1000, device)
    final_perplexity = np.exp(final_val_loss)
    writer.add_text(
        "summary/final_val_metrics",
        f"Full val loss: {final_val_loss:.4f}, Full val perplexity: {final_perplexity:.2f}, Number of samples: {1000 * batch_size:,}",
        step+1
    )
    print(f"⏱️ Full validation for {batch_size * 5000} samples is {final_val_loss:.4f}. Time={perf_counter() - t:.2f}s") 
    # close writer
    writer.close()

def main(config, logs_prefix):
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
    tokens_data = {
        "train": np.load(config["dataset_path"]["tokenized"], mmap_mode='r'),
        "validate": np.load(config["dataset_path"]["tokenized_validate"], mmap_mode='r')
    }
    print(f"batch_size = {batch_size} | warmup = {config["optimizer"]["scheduler"]["warmup_iters"]} | lr_max = {config["optimizer"]["lr"]} | lr_min = {config["optimizer"]["scheduler"]["lr_min"]}")
    lr_max, lr_min = config["optimizer"]["lr"], config["optimizer"]["scheduler"]["lr_min"]
    warmup = config["optimizer"]["scheduler"]["warmup_iters"]
    dataset_name = "TinyStories" # TODO: modify
    run_name = f"{dataset_name}/{logs_prefix}/steps_{steps}/warmup_{int(warmup * steps)}/exp_bs_{batch_size}/cosine_lrmax{lr_max}_lrmin{lr_min}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train_loop(steps, model, optimizer, sched_params, loss_fn, tokens_data, batch_size, context_length, model_params["device"], model_params["dtype"], optim_params["lr"], config["save_model"]["path"], run_name = run_name)



if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('--log-structure', type=str, default='base', help = 'subfolders structure inside run to log')
    inputs = parser.parse_args()
    print(inputs)

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    for lr in [1e-0, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]: # 1e+1
        config["optimizer"]["lr"] = lr
        for ratio in [1e-0, 1e-1, 1e-2, 1e-3]:
            config["optimizer"]["scheduler"]["lr_min"] = round(lr * ratio, 7)
            main(config, inputs.log_structure)
    
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
