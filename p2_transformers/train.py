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
from utils import *

seed = 123
torch.manual_seed(seed)

def train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config = None):
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
    num_tokens = tokens["train"].shape[0]
    
    # init indices
    if loader_mode == "sample":
        im_ids = None
    elif loader_mode == "in_memory_ids":
        t = perf_counter()
        # im_ids = np.arange(num_tokens - context_length, dtype = np.int32)
        # np.random.shuffle(im_ids)
        im_ids = np.random.choice(num_tokens - context_length, size=min(steps * os_bs, num_tokens), replace=False)
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

        # log validate params and log in tensorboard
        if step % valid_every == 0:
            valid_loss = eval_batch(tokens["valid"], model, loss_fn, context_length, batch_size, device)
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                if step / steps > serialize_first and step % serialize_every == 0:
                    save_checkpoint(model, optimizer, step, min_valid_loss, -1, ckpt_path, config)

            # add validate scalars to Tensorboard
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/perplexity", float('inf') if valid_loss > 20 else np.exp(valid_loss), step)

        # clip gradients TODO: add to parser
        if max_norm is not None:
            g_norm_pre, g_norm_post = gradient_clipping(model, max_l2_norm= max_norm)
            writer.add_scalar("train/grad_norm/post_clip", g_norm_post, step)
        else:
            g_norm_pre = compute_grad_norm(model)
        writer.add_scalar("train/grad_norm/pre_clip", g_norm_pre, step)
            
        
        # TODO: DELETE after testing trust_ratio
        param_to_name = {param: name for name, param in model.named_parameters()}
        optimizer.step(None, param_to_name) # TODO: DELETE
        # optimizer.step()
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
        save_checkpoint(model, optimizer, -1, -1, -1, ckpt_path, config)
    n_iter, loss_step, _, _ = load_checkpoint(ckpt_path, model, None, device)

    t = perf_counter()
    curr_time = datetime.now().time()
    print(f"Validation started at {curr_time.strftime('%H:%M:%S')}: Number of samples={valid_total}")

    final_valid_loss = eval(tokens["valid"], model, loss_fn, context_length, batch_size, valid_total, device)
    final_perplexity = np.exp(final_valid_loss)
    save_checkpoint(model, optimizer, n_iter, loss_step, final_valid_loss, ckpt_path, config)
    writer.add_text(
        "summary/final_valid_metrics",
        f"Full valid loss: {final_valid_loss:.4f}, Full valid perplexity: {final_perplexity:.2f}, Number of samples: {valid_total:,}, Time={perf_counter() - t:.2f}s",
        step+1
    )
    
    # close writer
    writer.close()

def main(config):
    # parse parameters
    model_params, ckpt_path, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    max_norm = clip_grad_params["max_norm"]

    # print intro
    curr_time = datetime.now().time()
    print()
    optim_suffix = "_tr" if config["optimizer"]["name"] in {"Lion"} and config["optimizer"]["is_trust_ratio"] else ""
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
        f"warmup={scheduler_params['warmup_iters']} | "
        f"z_alpha={run_params['z_alpha']} | "
    )

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
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Loader size: train={tokens['train'].shape[0]:,} | validate={tokens['valid'].shape[0]:,}")
    
    # create loss
    def loss_fn(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = run_params["z_alpha"]):
        return cross_entropy(logits, target, float(z_alpha))
    
    # run training loop
    train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config)
    print("-" * 100)

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    # config["optimizer"]["lr"] = 5e-2
    # config["optimizer"]["scheduler"]["warmup_iters"] = 0.01
    # config["train"]["optim_step_batch_size"] = 2560
    # config["optimizer"]["is_trust_ratio"] = True
    config["device"] = 1
    config["optimizer"]["lr"] = 1e-4
    config["optimizer"]["scheduler"]["warmup_iters"] = 0.05
    config["train"]["total_tokens_processed"] = 32_768_000 * 10 * 4
    main(config)
    # for norm_before, norm_after in [[None, "RMSNorm"]]: # [[None, "RMSNorm"], ["RMSNorm", None], ["RMSNorm", "RMSNorm"]]:
    #     for lr in [5e-4, 1e-4, 5e-5]:
    #         config["model"]["norms"]["before"] = norm_before
    #         config["model"]["norms"]["after"] = norm_after
    #         config["optimizer"]["lr"] = lr
    #         main(config)

    # TODO: remove it
    # config["optimizer"]["weight_decay"] = 5e-4
    # for lr_max, lr_min in [[3e-4, 3e-6], [3e-5, 3e-7]]: # [[5e-5, 5e-7], [5e-6, 5e-8]]: # 
    #     config["optimizer"]["lr"] = lr_max # 3e-3 # 1e-2 # 3e-4
    #     config["optimizer"]["scheduler"]["lr_min"] = lr_min  # 1e-3 # 3e-5
    #     config["train"]["optim_step_batch_size"] = 64
    #     for warmup_iters in [0.05, 0]:
    #         config["optimizer"]["scheduler"]["warmup_iters"] = warmup_iters
    #         main(config)
    # TODO: remove it


    # # testing AdamW
    # for weight_decay in [5e-4, 1e-4, 0]: # 1e-2, 1e-3, 
    #     config["optimizer"]["weight_decay"] = weight_decay
    #     for lr in [1e-3, 1e-4]:
    #         config["optimizer"]["lr"] = lr
    #         for lr_min in [1e-3, 1e-5, 1e-7]:
    #             config["optimizer"]["scheduler"]["lr_min"] = lr_min
    #             for optim_step_batch_size in [64]: #[1280, 64]:
    #                 config["train"]["optim_step_batch_size"] = optim_step_batch_size
    #                 main(config)

    # testing trust ratio in Lion
    # for lr in [6e-3]:
    #     config["optimizer"]["lr"] = lr
    #     config["optimizer"]["scheduler"]["lr_min"] = lr / 10
    #     for is_trust_ratio in [True]: # [False, True]:
    #         config["optimizer"]["is_trust_ratio"] = is_trust_ratio
    #         for optim_step_batch_size in [64]: #[1280, 64]:
    #             config["train"]["optim_step_batch_size"] = optim_step_batch_size
    #             main(config)
    
