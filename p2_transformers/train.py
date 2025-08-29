import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import yaml
from argparse import ArgumentParser
from pathlib import Path
import os
from time import perf_counter, sleep
from datetime import datetime
import numpy as np

from models import TransformerLM
from utils import * #parse_config, cross_entropy, cosine_lr_schedule, get_start_seqs, data_loading, get_valid_loss, gradient_clipping, save_checkpoint, load_checkpoint, get_short_gpu_name, get_optim


seed = 123
torch.manual_seed(seed)

def train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params):
    # unpack params
    steps = run_params["steps"]
    batch_size = run_params["batch_size"]
    os_bs = run_params["optimizer_step_batch_size"]
    context_length = run_params["context_length"]
    valid_every = max(1, int(1 / run_params["valid_freq"] * batch_size / os_bs)) # TODO: probably, it makes sense to divide by 'os_bs / batch_size'
    valid_total = run_params["valid_total"] if run_params["valid_total"] >= 0 else tokens["valid"].shape[0]
    serialize_path = run_params["serialize_path"]
    serialize_first = run_params["serialize_first"]
    serialize_every = max(1, int(1 / run_params["serialize_freq"] * batch_size / os_bs))  
    run_name = run_params["run_name"]
    device = run_params["device"]
    loader_mode = run_params["loader_mode"]
    lr = scheduler_params["lr_max"]
    
    # init indices
    if loader_mode == "sample":
        im_ids = None
    elif loader_mode == "in_memory_ids":
        t = perf_counter()
        im_ids = np.arange(tokens["train"].shape[0] - context_length, dtype = np.int32)
        np.random.shuffle(im_ids)
        print(f"⏱️ Created and shuffled in-memory ids: {im_ids.shape[0]:,} tokens. Time={perf_counter() - t:.2f}s") 

    # init logging and serialization
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    ckpt_path = Path(serialize_path) / run_name / "ckpt_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.train()
    loop = tqdm(range(steps), leave = True)
    valid_loss, min_train_loss, min_valid_loss = float('nan'), float('inf'), float('inf')

    # start training
    for step in loop:
        t_start = perf_counter()
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
            # start_seqs = get_start_seqs(step * batch_size, batch_size, tokens["train"].shape[0] - context_length, im_ids, loader_mode)
            tokens_curr, tokens_next = data_loading(tokens["train"], context_length, start_seqs, device)
            logits = model(tokens_curr)
            loss = loss_fn(logits, tokens_next)
            loss_acc +=  loss.item() / accum_steps
            (loss / accum_steps).backward()
        if loss_acc < min_train_loss:
            min_train_loss = loss_acc
        

        # log train params to tensorboard
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)
        writer.add_scalar("train/loss", loss_acc, step)
        writer.add_scalar("train/perplexity", float('inf') if loss_acc > 20 else np.exp(loss_acc), step)
        grad_norm = compute_grad_norm(model)
        writer.add_scalar("train/grad_norm_before_clip", grad_norm, step)

        # log validate params and log in tensorboard
        if step % valid_every == 0:
            valid_loss = eval_batch(tokens["valid"], model, loss_fn, context_length, batch_size, device)
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                if step / steps > serialize_first and step % serialize_every == 0:
                    save_checkpoint(model, optimizer, step, ckpt_path)

            # add validate scalars to Tensorboard
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/perplexity", float('inf') if valid_loss > 20 else np.exp(valid_loss), step)

        # clip gradients TODO: add to parser
        if max_norm is not None:
            gradient_clipping(model.parameters(), max_l2_norm= max_norm)
            grad_norm = compute_grad_norm(model)
            writer.add_scalar("train/grad_norm_after_clip", grad_norm, step)
        
        # TODO: DELETE
        # NOTE: test trust ratio
        param_to_name = {param: name for name, param in model.named_parameters()}
        # TODO: DELETE
        optimizer.step(None, param_to_name) # TODO: DELETE

        # log 'tokens per second'
        tokens_per_sec = os_bs * context_length / (perf_counter() - t_start)
        writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)

        # update progress bar
        loop.set_postfix(
            lr=f"{lr:.2e}", 
            train_loss=f"{loss_acc:.3f}", 
            min_train_loss=f"{min_train_loss:.3f}",
            valid_loss=f"{valid_loss:.3f}",
            min_valid_loss=f"{min_valid_loss:.3f}"
        )
    # run the best model on the full valid set
    if not ckpt_path.exists():
        save_checkpoint(model, optimizer, -1, ckpt_path)
    n_iter = load_checkpoint(ckpt_path, model, None)
    t = perf_counter()
    curr_time = datetime.now().time()
    print(f"Validation started at {curr_time.strftime('%H:%M:%S')}: Number of samples={valid_total}")

    final_valid_loss = eval(tokens["valid"], model, loss_fn, context_length, batch_size, valid_total, device)
    final_perplexity = np.exp(final_valid_loss)
    writer.add_text(
        "summary/final_valid_metrics",
        f"Full valid loss: {final_valid_loss:.4f}, Full valid perplexity: {final_perplexity:.2f}, Number of samples: {valid_total:,}, Time={perf_counter() - t:.2f}s",
        step+1
    )
    
    # close writer
    writer.close()

def main(config):
    # parse parameters
    model_params, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    max_norm = clip_grad_params["max_norm"]

    # print intro
    curr_time = datetime.now().time()
    print()
    optim_suffix = "_tr" if config["optimizer"]["is_trust_ratio"] else ""
    optim_name = config["optimizer"]["name"] + optim_suffix
    print(
        f"Run started at {curr_time.strftime("%H:%M:%S")}: bs={run_params['batch_size']} | "
        f"optim_step_bs={run_params['optimizer_step_batch_size']} | "
        f"lr_max={clean(scheduler_params['lr_max'])} | "
        f"lr_min={clean(scheduler_params['lr_min'])} | "
        f"w_decay={clean(optimizer_params['weight_decay'])} | "
        f"context={run_params['context_length']} | "
        f"device={get_short_gpu_name(config["device"])} | "
        f"optim={optim_name} | "
        f"steps={run_params['steps']} | "
        f"warmup={scheduler_params['warmup_iters']}"
    )

    # get experiments
    model = TransformerLM(**model_params).to(model_params["device"]) # NOTE: I sent to device explicitely. Not sure if it makes sense.
    optimizer_params["params"] = model.parameters()
    optimizer = get_optim(config["optimizer"]["name"], optimizer_params)
    tokens = {
        "train": np.load(tokens_params["train"], mmap_mode='r'),
        "valid": np.load(tokens_params["valid"], mmap_mode='r')
    }
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Loader size: train={tokens['train'].shape[0]:,} | validate={tokens['valid'].shape[0]:,}")
    loss_fn = cross_entropy # TODO: probably, it makes sense to automate
    
    # run training loop
    train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params)
    print("-" * 100)

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    # parser.add_argument('--log-structure', type=str, default='rtx4000ada', help = 'subfolders structure inside run to log')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    for lr in [6e-3]: # TODO: add 6e-3
        config["optimizer"]["lr"] = lr
        config["optimizer"]["scheduler"]["lr_min"] = lr / 10
        for is_trust_ratio in [False, True]:
            config["optimizer"]["is_trust_ratio"] = is_trust_ratio
            for optim_step_batch_size in [1280, 64]:
                config["train"]["optim_step_batch_size"] = optim_step_batch_size
                main(config)
    # for lr_max in [5e-3, 1e-3]: # 1e+1, 1e-0, 1e-1, 5e-2, 1e-2, 1e-4
    
    # normal Lion
    # config["train"]["batch_size"] = 64
    # for lr in [1e-3, 5e-4, 3e-4,  1e-4]:
    #     config["optimizer"]["lr"] = lr
    #     config["optimizer"]["scheduler"]["lr_min"] = lr * 1e-2
    #     main(config)
        
    # Lion+trust_ratio
    # config["optimizer"]["is_trust_ratio"] = True
    # for bs in [64, 128, 192]:
    #     config["train"]["batch_size"] = bs
    #     for lr_max in [5e-3, 3e-3, 1e-3, 5e-4, 3e-4,  1e-4]:
    #         config["optimizer"]["lr"] = lr_max
    #         # for lr_min in [1e-3, 1e-4, 1e-5, 1e-6]: # , 1e-7
    #         for ratio in [1e-1, 1e-2]:
    #             lr_min = ratio * lr_max
    #             config["optimizer"]["scheduler"]["lr_min"] = lr_min
    #             main(config)
    
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
