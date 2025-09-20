import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
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

def train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config = None, flops_per_token: int = 0):
    # unpack params
    steps = run_params["steps"]
    eval_steps = {int(r * steps) for r in config["validate"]["eval_steps"]}
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
    autocast_dtype = run_params["autocast_dtype"]
    debug = config["debug"]
    logger_name = run_params["logger_name"]
    
    # init indices
    if loader_mode == "sample":
        im_ids = None
    elif loader_mode == "in_memory_ids":
        t = perf_counter()
        size_to_sample = num_tokens - context_length
        im_ids = np.random.choice(size_to_sample, size=min(steps * os_bs, size_to_sample), replace=False)
        print(f"Sampled to keep in-memory ids: {im_ids.shape[0]:,} tokens. Time={perf_counter() - t:.2f}s") 

    # init logging and serialization
    if not debug:
        if logger_name == "wandb":
            wandb.init(project = "llms-from-scratch", name = run_name, config = config)
        else:
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
        logits_norm_acc = 0.0
        logits_max_acc = 0.0
        logits_std_acc = 0.0
        accum_steps = os_bs // batch_size
        for i in range(accum_steps):
            start_seqs = get_start_seqs(step * os_bs + i * batch_size, batch_size, tokens["train"].shape[0] - context_length, im_ids, loader_mode)
            tokens_curr, tokens_next = data_loading(tokens["train"], context_length, start_seqs, device)
            with autocast('cuda', enabled = is_amp, dtype=autocast_dtype):
                logits = model(tokens_curr)
                loss = loss_fn(logits, tokens_next)
            
            # logging in the base dtype (fp32)
            loss_acc +=  loss.item() / accum_steps
            
            # scale in the base dtype (fp32)
            loss /= accum_steps
            if is_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Logit statistics (no grad)
            with torch.no_grad():
                logits_norm_acc += logits.norm(dim=-1).mean().item() / accum_steps
                logits_max_acc  += logits.amax(dim=-1).mean().item() / accum_steps
                logits_std_acc  += logits.std().item() / accum_steps

        if loss_acc < min_train_loss:
            min_train_loss = loss_acc
            # Reduce interval between heavy evaluations near the end
            if steps - (step + 1) <= near_end_threshold:
                serialize_min_steps = config["serialize"]["min_steps"]
            # Check if we should run heavy eval
            if not debug and step - heavy_val_step >= serialize_min_steps:
                heavy_val_loss = eval(step, tokens["valid"], model, loss_fn, context_length, batch_size, heavy_valid_total, device, False, use_amp = is_amp)
                heavy_vals_num += 1
                if  heavy_val_loss < min_valid_loss:
                    min_valid_loss = heavy_val_loss
                    save_checkpoint(model, optimizer, step, min_valid_loss, -1, ckpt_path, config)
                    heavy_val_step = step

        # log train params to tensorboard
        if not debug:
            if logger_name == "wandb":
                wandb.log({
                    "lr": optimizer.param_groups[0]['lr'],
                    "train/loss": loss_acc,
                    "train/perplexity": float('inf') if loss_acc > 20 else np.exp(loss_acc),
                    "train/logits_norm": logits_norm_acc,
                    "train/logits_max": logits_max_acc,
                    "train/logits_std": logits_std_acc,
                }, step=step)
            else:
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)
                writer.add_scalar("train/loss", loss_acc, step)
                writer.add_scalar("train/perplexity", float('inf') if loss_acc > 20 else np.exp(loss_acc), step)
                writer.add_scalar("train/logits_norm", logits_norm_acc, step)
                writer.add_scalar("train/logits_max", logits_max_acc, step)
                writer.add_scalar("train/logits_std", logits_std_acc, step)
            

            # log validate params and log in tensorboard
            if step % valid_every == 0:
                valid_loss = eval_batch(tokens["valid"], model, loss_fn, context_length, batch_size, device)
                if logger_name == "wandb":
                    wandb.log({
                        "valid/loss": valid_loss,
                        "valid/perplexity": float('inf') if valid_loss > 20 else np.exp(valid_loss),
                    }, step=step)
                else:
                    writer.add_scalar("valid/loss", valid_loss, step)
                    writer.add_scalar("valid/perplexity", float('inf') if valid_loss > 20 else np.exp(valid_loss), step)

        # clip gradients
        if max_norm is not None:
            if is_amp:
                scaler.unscale_(optimizer)
            g_norm_pre, g_norm_post = gradient_clipping(model, max_l2_norm= max_norm)
        else:
            g_norm_pre = compute_grad_norm(model)
            g_norm_post = None
        if not debug:
            if logger_name == "wandb":
                log_data = {"train/grad_norm/pre_clip": g_norm_pre}
                if g_norm_post is not None:
                    log_data["train/grad_norm/post_clip"] = g_norm_post
                wandb.log(log_data, step=step)
            else:
                writer.add_scalar("train/grad_norm/pre_clip", g_norm_pre, step)
                if g_norm_post is not None:
                    writer.add_scalar("train/grad_norm/post_clip", g_norm_post, step)
            
        if is_amp: # NOTE: it will not work with Lion_tr
            scaler.step(optimizer)
            scaler.update()
        else:
            # TODO: DELETE after testing trust_ratio
            param_to_name = {param: name for name, param in model.named_parameters()}
            if config["optimizer"]["name"] == "Lion":
                optimizer.step(None, param_to_name) # TODO: DELETE
            else:
                optimizer.step()
        if not debug:
            # log 'tokens per second'
            tokens_per_sec = os_bs * context_length / (perf_counter() - t_step)
            flops_per_sec = flops_per_token * os_bs * context_length / (perf_counter() - t_step)
            if logger_name == "wandb":
                wandb.log({
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/forward_flops_per_sec": flops_per_sec,
                }, step=step)
            else:
                writer.add_scalar("train/tokens_per_sec", tokens_per_sec, step)
                writer.add_scalar("train/forward_flops_per_sec", flops_per_sec, step)

        # run eval of large subset of train and valid datasets if asked
        if not debug and step + 1 in eval_steps and step + 1 != steps:
            summary = log_evals(ckpt_path, step, t_train, tokens, model, optimizer, loss_fn, config, use_amp = is_amp)
            if logger_name == "wandb":
                wandb.run.summary[f"summary_at_step_{step + 1}"] = summary
            else:
                writer.add_text("summary/final_valid_metrics", summary, step+1)

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
    if not debug:
        summary = log_evals(ckpt_path, step, t_train, tokens, model, optimizer, loss_fn, config, use_amp = is_amp)
        if logger_name == "wandb":
                wandb.run.summary[f"summary_at_step_{step + 1}"] = summary
                wandb.finish()
        else:
            writer.add_text("summary/final_valid_metrics", summary, step+1)
            writer.close()            

def main(config, random_seed = 123):
    # control source of randonmess
    torch.manual_seed(seed)

    # parse parameters
    model_params, ckpt_path, optimizer_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    config["device_name"] = run_params["device_name"]
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
        f"{colored('Dataset: ', 'blue', attrs=["bold"])}{dataset_name} | {colored('bs=', 'blue')}{run_params['batch_size']:,} | "
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
        f"{colored('w_tying=', 'blue')}{config['model']['weights_tying']} | " 
    )
    print_d_model_d_ff(model_params["d_model"], model_params["d_ff"], model_params['is_gate'])
    
    # FLOPS estimations
    flops_per_token = est_forward_flops(config)
    total_tokens = config["train"]["total_tokens_processed"]
    print(f"{colored('Estimated number of FLOPS: ', 'blue')} per_token={flops_per_token:,} | total={flops_per_token * total_tokens:.2e}")

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
    # details of the model
    cnt_params = count_parameters(model)
    print(
        f"{colored("Model parameters: ", 'blue')}{cnt_params:,} | "
        f"{colored('num_layers=', 'blue')}{model_params["num_layers"]:,} | "
        f"{colored('num_heads=', 'blue')}{model_params["num_heads"]:,} | "
        f"{colored('d_model=', 'blue')}{model_params["d_model"]:,} | "
        f"{colored('d_ff=', 'blue')}{model_params["d_ff"]:,} |"
        f"{colored('Ratio tokens/model_size=', 'blue')}{total_tokens/cnt_params:.2f} |"
    )
    memories_stat = get_expected_memory(config)
    print_memory_stats(memories_stat)

    # details of the loader
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
    train_loop(model, optimizer, tokens, loss_fn, scheduler_params, max_norm, run_params, config, flops_per_token)
    print(colored("-" * 200, "cyan", attrs=["bold"]))

if __name__ == '__main__':
    curr_time = datetime.now().time()
    print(colored(f"⏱️ Run started at {curr_time.strftime("%H:%M:%S")}.", 'red', attrs=["bold"]))
    print(colored("-" * 200, 'red', attrs=["bold"]))
    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)

    config["serialize"]["postfix"] = "winit2e-2_xav"

    def update_config(config, mode = 'baseline'):
        if mode == 'test':
            config["train"]["batch_size"] = 32 # NOTE: only for RTX5090
            config["model"]["context_length"] = 352
            config["model"]["d_model"] = 1024
            config["model"]["d_ff"] = 2688
            config["model"]["weights_tying"] = True
            config["model"]["num_layers"] = 16
            config["model"]["num_heads"] = 16
            config["model"]["norms"] = {
                'before': 'RMSNorm',
                'after': None,
                'residual': None,
                'final': 'RMSNorm'
            }
            config["optimizer"]["name"] = 'AdamW'
            config["optimizer"]["lr"] = 5e-3
            config["optimizer"]["beta1"] = 0.9
            config["optimizer"]["beta2"] = 0.98
    
            config["optimizer"]["scheduler"]["lr_min"] = 3e-5
            config["optimizer"]["scheduler"]["warmup_iters"] = 2000
            config["optimizer"]["scheduler"]["cosine_cycle_iters"] = 1.0
        
        return config


    # number of FLOPS and token count
    # flops_per_token = est_forward_flops(config)
    # steps =  int(4 * 10**16 / flops_per_token / config["train"]["optim_step_batch_size"] / config["model"]["context_length"])
    # config["train"]["total_tokens_processed"] = steps * config["train"]["optim_step_batch_size"] * config["model"]["context_length"]
    # config["optimizer"]["scheduler"]["warmup_iters"] = min(max(50, int(0.05 * steps)), 100)

    # config["device"] = 1
    # config = update_config(config, 'test')
    config["train"]["total_tokens_processed"] //= 2
    for init_type in ["xavier", "lecun"]:
        config["model"]["inits"]["type_ff"] = init_type    
        for std in [1.0, 0.02]:
            config["model"]["inits"]["std_emb"] = std
            main(config, seed)