import torch
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from tqdm import tqdm
import yaml
from argparse import ArgumentParser
from pathlib import Path
from termcolor import colored
import os
from time import perf_counter, sleep
from datetime import datetime
import numpy as np

from models import TransformerLM
from utils import *

seed = 123
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')

def train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config = None):
    # unpack params
    steps = run_params["steps"]
    batch_size = run_params["batch_size"]
    os_bs = run_params["optimizer_step_batch_size"]
    context_length = run_params["context_length"]
    valid_every = max(1, int(1 / run_params["valid_freq"] * batch_size / os_bs)) # TODO: probably, it makes sense to divide by 'os_bs / batch_size'
    valid_total = run_params["valid_total"] if run_params["valid_total"] >= 0 else tokens["valid"].shape[0]
    heavy_valid_total = config["serialize"]["num_samples"]
    serialize_path = run_params["serialize_path"]
    serialize_min_steps = config["serialize"]["steps"]
    near_end_threshold = config["serialize"]["near_end_threshold"]
    serialize_every = max(1, int(1 / run_params["serialize_freq"] * batch_size / os_bs))  
    run_name = run_params["run_name"]
    device = run_params["device"]
    loader_mode = run_params["loader_mode"]
    lr = scheduler_params["lr_max"]
    num_tokens = tokens["train"].shape[0]
    is_amp = config["model"]["dtype"] == "amp"
    
    # init indices
    if loader_mode == "sample":
        im_ids = None
    elif loader_mode == "in_memory_ids":
        t = perf_counter()
        im_ids = np.random.choice(num_tokens - context_length, size=min(steps * os_bs, num_tokens), replace=False)
        print(f"Sampled to keep in-memory ids: {im_ids.shape[0]:,} tokens. Time={perf_counter() - t:.2f}s") 

    # init logging and serialization
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    ckpt_path = Path(serialize_path) / run_name / "ckpt_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.train()
    loop = tqdm(range(steps), leave = True, desc = colored("Training", 'blue', attrs=["bold"]))
    valid_loss, min_train_loss, min_valid_loss = float('nan'), float('inf'), float('inf')

    # start training
    t_train = perf_counter()
    scaler = GradScaler('cuda') if is_amp else None
    heavy_val_step, heavy_vals_num = -1, 0
    for step in loop:
        t_step = perf_counter()
        # update learning rate TODO: create one-line function
        scheduler_params["t"] = step + 1
        lr = cosine_lr_schedule(**scheduler_params)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
           
        optimizer.zero_grad()
        loss_acc = 0
        accum_steps = os_bs // batch_size
        for i in range(accum_steps):
            start_seqs = get_start_seqs(step * os_bs + i * batch_size, batch_size, tokens["train"].shape[0] - context_length, im_ids, loader_mode)
            tokens_curr, tokens_next = data_loading(tokens["train"], context_length, start_seqs, device)
            if is_amp:
                with autocast('cuda'):
                    logits = model(tokens_curr)
                    loss = loss_fn(logits, tokens_next)
                    loss_acc +=  loss.item() / accum_steps
                    scaler.scale(loss / accum_steps).backward()
            else:
                logits = model(tokens_curr)
                loss = loss_fn(logits, tokens_next)
                loss_acc +=  loss.item() / accum_steps
                (loss / accum_steps).backward()

        if loss_acc < min_train_loss:
            min_train_loss = loss_acc
            # Reduce interval between heavy evaluations near the end
            if steps - (step + 1) <= near_end_threshold:
                serialize_min_steps = config["serialize"]["min_steps"]
            # Check if we should run heavy eval
            if step - heavy_val_step >= serialize_min_steps:
                heavy_val_loss = eval(step, tokens["valid"], model, loss_fn, context_length, batch_size, heavy_valid_total, device, False, use_amp = is_amp)
                heavy_vals_num += 1
                if  heavy_val_loss < min_valid_loss:
                    min_valid_loss = heavy_val_loss
                    save_checkpoint(model, optimizer, step, min_valid_loss, -1, ckpt_path, config)
                    heavy_val_step = step

        # log train params to tensorboard
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        writer.add_scalar("train/loss", loss_acc, step)
        writer.add_scalar("train/perplexity", float('inf') if loss_acc > 20 else np.exp(loss_acc), step)

        # log validate params and log in tensorboard
        if step % valid_every == 0:
            valid_loss = eval_batch(tokens["valid"], model, loss_fn, context_length, batch_size, device)
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/perplexity", float('inf') if valid_loss > 20 else np.exp(valid_loss), step)

        # clip gradients
        if max_norm is not None:
            if is_amp:
                scaler.unscale_(optimizer)
            g_norm_pre, g_norm_post = gradient_clipping(model, max_l2_norm= max_norm)
            writer.add_scalar("train/grad_norm/post_clip", g_norm_post, step)
        else:
            g_norm_pre = compute_grad_norm(model)
        writer.add_scalar("train/grad_norm/pre_clip", g_norm_pre, step)
            
        
        if is_amp: # NOTE: it will not work with Lion_tr
            scaler.step(optimizer)
            scaler.update()
        else:
            # TODO: DELETE after testing trust_ratio
            param_to_name = {param: name for name, param in model.named_parameters()}
            optimizer.step(None, param_to_name) # TODO: DELETE

        # optimizer.step()
        # log 'tokens per second'
        tokens_per_sec = os_bs * context_length / (perf_counter() - t_step)
        writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

        # run eval of large subset of train and valid datasets if asked
        if step + 1 in config["validate"]["eval_steps"] and step + 1 != steps:
            log_evals(ckpt_path, step, t_train, tokens, model, optimizer, loss_fn, config, writer, use_amp = is_amp)

        # update progress bar
        loop.set_postfix(
            lr=f"{lr:.2e}", 
            train_loss=f"{loss_acc:.3f}", 
            min_train_loss=f"{min_train_loss:.3f}",
            valid_loss=f"{valid_loss:.3f}",
            min_valid_loss=f"{min_valid_loss:.3f}",
            heavy_vals = f"{heavy_vals_num}"
        )
    # run the best model on the large subsets of train and valid set
    log_evals(ckpt_path, step, t_train, tokens, model, optimizer, loss_fn, config, writer, use_amp = is_amp)
    
    # close writer
    writer.close()

def main(config):
    # parse parameters
    model_params, ckpt_path, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    max_norm = clip_grad_params["max_norm"]

    # print intro
    curr_time = datetime.now().time()
    print()
    print(colored("-" * 200, "cyan", attrs=["bold"]))
    optim_suffix = "_tr" if config["optimizer"]["name"] in {"Lion"} and config["optimizer"]["is_trust_ratio"] else ""
    optim_name = config["optimizer"]["name"] + optim_suffix
    dataset_name = Path(config["dataset_path"]["prefix"]).name
    print(colored(f"⏱️ Experiment started at {curr_time.strftime("%H:%M:%S")}.", 'blue', attrs=["bold"]))
    warmup_perc = 100 * scheduler_params['warmup_iters'] / run_params['steps']
    print(
        f"{colored('Dataset: ', 'blue', attrs=["bold"])}{dataset_name} | bs={run_params['batch_size']:,} | "
        f"{colored('optim_step_bs=', 'blue')}{run_params['optimizer_step_batch_size']:,} | "
        f"{colored('context=', 'blue')}{run_params['context_length']} | "
        f"{colored('optim=', 'blue')}{optim_name} | "
        f"{colored('steps=', 'blue')}{run_params['steps']:,} | "
        f"{colored('warmup=', 'blue')}{scheduler_params['warmup_iters']:,} ({warmup_perc:.2f}% of total) | "
        f"{colored('lr_max=', 'blue')}{clean(scheduler_params['lr_max'])} | "
        f"{colored('lr_min=', 'blue')}{clean(scheduler_params['lr_min'])} | "
        f"{colored('w_decay=', 'blue')}{clean(optimizer_params['weight_decay'])} | "
        f"{colored('z_alpha=', 'blue')}{clean(run_params['z_alpha'])} | "        
        f"{colored('device=', 'blue') }{get_short_gpu_name(config["device"])} | "
        f"{colored('activation=', 'blue')}{'Gated ' if model_params['is_gate'] else ''}{model_params['activation']} | "
        f"{colored('dtype=', 'blue')}{config['model']['dtype']} | "
        f"{colored('compile=', 'blue')}{config['train']['compile']} | " 
    )
    print_d_model_d_ff(model_params["d_model"], model_params["d_ff"], model_params['is_gate'])

    # get experiments
    model = TransformerLM(**model_params).to(model_params["device"]) # NOTE: I sent to device explicitely. Not sure if it makes sense.
    if ckpt_path:
        n_iter, loss_step, loss_full, _ = load_checkpoint(ckpt_path, model, None, model_params["device"])
        run_params["run_name"] = run_params["run_name"].format(f"{loss_full:.4f}" if loss_full else "unkn")
    optimizer_params["params"] = model.parameters()
    optimizer = get_optim(config["optimizer"]["name"], optimizer_params)
    tokens = {
        "train": np.load(tokens_params["train"], mmap_mode='r'),
        "valid": np.load(tokens_params["valid"], mmap_mode='r')
    }
    print(
        colored("Model parameters: ", 'blue') + f"{count_parameters(model):,} | "
        f"{colored('num_layers=', 'blue')}{model_params["num_layers"]:,} | "
        f"{colored('num_heads=', 'blue')}{model_params["num_heads"]:,} | "
        f"{colored('d_model=', 'blue')}{model_params["d_model"]:,} | "
        f"{colored('d_ff=', 'blue')}{model_params["d_ff"]:,} |"
    )
    print(
        f"{colored("Loader size: ", 'blue')} "
        f"{colored('train=', 'blue')}{tokens['train'].shape[0]:,} | "
        f"{colored('validate=', 'blue')}{tokens['valid'].shape[0]:,} | "
        f"{colored('Tokens to process=', 'blue')}{config['train']['total_tokens_processed']:,} | "
    )
    
    # create loss
    def loss_fn(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = run_params["z_alpha"]):
        return cross_entropy(logits, target, float(z_alpha))
    
    # run training loop
    if config["train"]["compile"]:
        model = torch.compile(model)
    train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config)
    print(colored("-" * 200, "cyan", attrs=["bold"]))

if __name__ == '__main__':
    seed = 124
    torch.manual_seed(seed)

    curr_time = datetime.now().time()
    print(colored(f"⏱️ Run started at {curr_time.strftime("%H:%M:%S")}.", 'red', attrs=["bold"]))
    print(colored("-" * 200, 'red', attrs=["bold"]))
    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)

    config["device"] = 1
    config["validate"]["eval_steps"] = {5_000, 10_000}
    
    # # NOTE: TEST!!!
    config["model"]["activation"] = "GELU"
    config["model"]["is_gate"] = False
    config["optimizer"]["name"] = 'AdamW'
    config["optimizer"]["lr"] = 3e-3
    config["optimizer"]["beta2"] = 0.98
    config["optimizer"]["weight_decay"] = 0.01
    config["optimizer"]["scheduler"]["lr_min"] = 1e-3
    config["train"]["optimizer_step_batch_size"] = 256
    # # NOTE: TEST!!!
    
    # config["model"]["is_gate"] = True
    # config["model"]["d_ff"] = 2016 if config["model"]["is_gate"] else 3072
    config["model"]["d_ff"] = 4032 // 3 if config["model"]["is_gate"] else 6144 // 3

    # lr = 1e-4
    # config["optimizer"]["lr"] = lr
    # config["optimizer"]["scheduler"]["lr_min"] = lr / 10
    for num_heads in [16]: # 
        config["model"]["num_heads"] = num_heads
        main(config)