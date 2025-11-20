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

from p1_core.models import TransformerLM
from p1_core.utils import *


def train_loop(n_iter,  model, optimizer, scaler, tokens, loss_fn, scheduler_params, max_norm, run_params, config = None, flops_per_token: int = 0):
    # unpack params
    steps = run_params["steps"]
    eval_steps = {int(r * steps) for r in config["validate"]["report"]["steps_ratio"]}
    bs = run_params["bs"]
    os_bs = run_params["os_bs"]
    context_length = run_params["context_length"]
    valid_every = max(1, int(1 / run_params["valid_freq"] * bs / os_bs))
    valid_ckpt_nums = config["validate"]["checkpoint"]["num_samples"]
    valid_rprt_nums = config["validate"]["report"]["num_samples"]
    serialize_path = run_params["serialize_path"]
    serialize_min_steps = config["validate"]["checkpoint"]["min_interval_steps"]
    serialize_max_steps = float('inf')
    near_end_threshold = config["validate"]["checkpoint"]["near_end"]["threshold"]
    run_name = run_params["run_name"]
    device = run_params["device"]
    loader_mode = run_params["loader_mode"]
    lr = scheduler_params["lr_max"]
    train_token_range = tokens["train"].shape[0] - context_length
    valid_token_range = tokens["valid"].shape[0] - context_length
    is_amp = config["model"]["dtype"] == "amp"
    autocast_dtype = run_params["autocast_dtype"]
    debug = config["debug"]
    logger_name = run_params["logger_name"]
    
    # Init indices for training
    num_train_ckpt_report = (steps * os_bs, valid_ckpt_nums, valid_rprt_nums)
    im_ids_train, im_ids_valid = sample_indices(loader_mode, train_token_range, valid_token_range, num_train_ckpt_report)
    
    # Init logging and serialization
    if not debug:
        if logger_name == "wandb":
            wandb.init(project = "llms-from-scratch", name = run_name, config = config)
        writer = SummaryWriter(log_dir=f"runs/{run_name}") if logger_name == "tensorboard" else None
        ckpt_save_path = Path(serialize_path) / run_name / "ckpt_best.pt"
        ckpt_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.train()
    loop = tqdm(
        range(n_iter, steps),
        leave = True,
        desc = colored("Training", 'blue', attrs=["bold"]),
        initial = n_iter,
        total = steps
    )
    valid_loss, min_train_loss, min_valid_loss = float('nan'), float('inf'), float('inf')

    # start training
    t_train = perf_counter()
    valid_ckpt_step, valid_ckpt_cnt = -1, 0
    valid_ckpt_summary = []
    for step in loop:
        t_step = perf_counter()
        
        # reset max stats for GPU
        if not debug:
            log_data = dict()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                log_gpu_stats(log_data, config, "start")

        # update learning rate TODO: create one-line function
        scheduler_params["t"] = step + 1
        lr = cosine_lr_schedule(**scheduler_params)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
           
        optimizer.zero_grad()
        loss_acc = 0
        logits_norm_acc, logits_max_acc, logits_std_acc = 0.0, 0.0, 0.0
        accum_steps = os_bs // bs
        for i in range(accum_steps):
            start_seqs = get_start_seqs(step * os_bs + i * bs, bs, train_token_range, im_ids_train, loader_mode)
            tokens_curr, tokens_next = data_loading(tokens["train"], context_length, start_seqs, device)
            with autocast('cuda', enabled = is_amp, dtype=autocast_dtype):
                logits = model(tokens_curr)
                loss = loss_fn(logits, tokens_next)
            
            # Logging in the base dtype (fp32)
            loss_acc +=  loss.item() / accum_steps
            
            # Scale in the base dtype (fp32)
            loss /= accum_steps

            # Perform optimizer step (AMP or non-AMP)
            if is_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Logit statistics (no grad)
            with torch.no_grad():
                logits_norm_acc += logits.norm(dim=-1).mean().item() / accum_steps
                logits_max_acc  += logits.amax(dim=-1).mean().item() / accum_steps
                logits_std_acc  += logits.std().item() / accum_steps

        # add memory statistics
        if not debug and torch.cuda.is_available():
            log_gpu_stats(log_data, config, "pass")

        # Reduce interval between validation for checkpointing near the end
        if steps - (step + 1) <= near_end_threshold:
            serialize_min_steps = config["validate"]["checkpoint"]["near_end"]["min_steps"]
            serialize_max_steps = config["validate"]["checkpoint"]["near_end"]["max_steps"]
        
        if loss_acc < min_train_loss or step - valid_ckpt_step >= serialize_max_steps or step == steps - 1:
            min_train_loss = min(loss_acc, min_train_loss)

            # Check if we should run eval for checkpointing
            if not debug and (step - valid_ckpt_step >= serialize_min_steps or step == steps - 1):
                valid_ckpt_loss = eval(step, tokens["valid"], model, loss_fn, context_length, bs, im_ids_valid["valid_ckpt"], device, False, use_amp = is_amp)
                valid_ckpt_cnt += 1
                valid_ckpt_summary.append(f"Step {step:,}: ❌")
                if  valid_ckpt_loss < min_valid_loss:
                    min_valid_loss = valid_ckpt_loss
                    save_checkpoint(model, optimizer, scaler, scheduler_params, step, min_valid_loss, -1, ckpt_save_path, config)
                    valid_ckpt_step = step
                    valid_ckpt_summary[-1] = f"Step {step:,}: ✅"
                if step - valid_ckpt_step >= serialize_max_steps:
                    valid_ckpt_step = step

        # update log data (train and validate params and stats)
        if not debug:
            log_train_stats(log_data, optimizer.param_groups[0]['lr'], loss_acc, logits_norm_acc, logits_max_acc, logits_std_acc)
            if step % valid_every == 0:
                valid_loss = eval_batch(tokens["valid"], model, loss_fn, context_length, bs, device)
                log_valid_stats(log_data, valid_loss)

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
                log_grad_norm_stats(log_data, g_norm_pre, g_norm_post)
            
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
            log_perf_stats(log_data, tokens_per_sec, flops_per_sec)

            # add memory statistics
            if torch.cuda.is_available():
                log_gpu_stats(log_data, config, "optim")       
                
        # run eval of large subset of train and valid datasets if asked
        if not debug and step + 1 in eval_steps and step + 1 != steps:
            summary = log_evals(ckpt_save_path, step, t_train, tokens, model, optimizer, scaler, scheduler_params, loss_fn, im_ids_valid, config, use_amp = is_amp)
            if logger_name == "wandb":
                wandb.run.summary[f"summary_at_step_{step+1}"] = summary
            else:
                writer.add_text("summary/final_valid_metrics", summary, step+1)

        # log stats
        if not debug:
            log_stats(log_data, step, logger_name, writer)
        # update progress bar
        loop.set_postfix(
            lr=f"{lr:.2e}", 
            train_loss=f"{loss_acc:.3f}", 
            min_train_loss=f"{min_train_loss:.3f}",
            valid_loss=f"{valid_loss:.3f}",
            min_valid_loss=f"{min_valid_loss:.3f}",
            valid_ckpt_cnt = f"{valid_ckpt_cnt}"
        )
    # run the best model on the large subsets of train and valid set
    if not debug:
        summary = log_evals(ckpt_save_path, step, t_train, tokens, model, optimizer, scaler, scheduler_params, loss_fn, im_ids_valid, config, use_amp = is_amp)
        valid_ckpt_summary_log = "\n".join(valid_ckpt_summary)
        if logger_name == "wandb":
                wandb.run.summary[f"summary_at_step_{step + 1}"] = summary
                wandb.run.summary["validation_checkpoints"] = valid_ckpt_summary_log
                wandb.finish()
        else:
            writer.add_text("summary/final_valid_metrics", summary, step+1)
            writer.add_text("summary/validation_checkpoints", valid_ckpt_summary_str, step + 1)
            writer.close()        

def main(config, random_seed = 123):
    # control source of randonmess
    torch.manual_seed(random_seed)
    torch.set_float32_matmul_precision('high')

    # parse parameters
    model_params, ckpt_path_load, optim_params, scheduler_params, clip_grad_params, tokens_params, run_params = parse_config(config)
    config["device_name"] = run_params["device_name"]
    max_norm = clip_grad_params["max_norm"]
    resume_training = "ckpt_load_from" in config["train"] and run_params["resume_training"]

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
        f"{colored('Dataset: ', 'blue', attrs=["bold"])}{dataset_name} | {colored('bs=', 'blue')}{run_params['bs']:,} | "
        f"{colored('optim_step_bs=', 'blue')}{run_params['os_bs']:,} | "
        f"{colored('context=', 'blue')}{run_params['context_length']} | "
        f"{colored('optim=', 'blue')}{optim_name} | "
        f"{colored('steps=', 'blue')}{run_params['steps']:,} | "
        f"{colored('warmup=', 'blue')}{scheduler_params['warmup_iters']:,} ({warmup_perc:.2f}% of total) | "
        f"{colored('lr_max=', 'blue')}{clean(scheduler_params['lr_max'])} | "
        f"{colored('lr_min=', 'blue')}{clean(scheduler_params['lr_min'])} | "
        f"{colored('w_decay=', 'blue')}{clean(optim_params['weight_decay'])} | "
        f"{colored('z_alpha=', 'blue')}{clean(config['loss'].get('z_alpha', 0.0))} | "        
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
    if config["train"]["compile"]:
        model = torch.compile(model)

    optim_params["params"] = model.parameters()
    optimizer = get_optim(config["optimizer"]["name"], optim_params)

    is_amp = config["model"]["dtype"] == "amp"
    scaler = GradScaler('cuda') if is_amp else None
    
    print("ckpt_path_load", ckpt_path_load)
    if ckpt_path_load is not None:
        # Resume full training state (model + optimizer + scaler + scheduler)
        if resume_training:
            n_iter, loss_ckpt, loss_report, scheduler_params = load_checkpoint(ckpt_path_load, model, optimizer, scaler, model_params["device"])
        else:
            # Only load pretrained weights for fine-tuning or evaluation
            n_iter, loss_ckpt, loss_report, _ = load_checkpoint(ckpt_path_load, model, None, None, model_params["device"])
        run_params["run_name"] = run_params["run_name"].format(f"{loss_report:.4f}" if loss_report else "unkn")

    tokens = {
        "train": np.load(tokens_params["train"], mmap_mode='r'),
        "valid": np.load(tokens_params["valid"], mmap_mode='r')
    }
   
    # add ratio total_tokens to params number
    cnt_params = count_parameters(model)
    config["train"]["tokens_model_ratio"] = total_tokens / cnt_params
     # details of the model
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

    # Details of the loader
    print(
        f"{colored("Loader size: ", 'blue')} "
        f"{colored('train=', 'blue')}{tokens['train'].shape[0]:,} | "
        f"{colored('validate=', 'blue')}{tokens['valid'].shape[0]:,} | "
        f"{colored('Tokens to process=', 'blue')}{config['train']['total_tokens_processed']:,} | "
    )
    
    # Create loss
    loss_fn = get_loss_fn(config["loss"])
    
    # Run training loop
    train_loop(n_iter if resume_training else 0, model, optimizer, scaler, tokens, loss_fn, scheduler_params, max_norm, run_params, config, flops_per_token)
    print(colored("-" * 200, "cyan", attrs=["bold"]))

if __name__ == '__main__':
    curr_time = datetime.now().time()
    print(colored(f"⏱️ Run started at {curr_time.strftime("%H:%M:%S")}.", 'red', attrs=["bold"]))
    print(colored("-" * 200, 'red', attrs=["bold"]))
    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='p1_core/configs/train_owt.yaml', help='config file')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    inputs = parser.parse_args()

    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)


    main(config, inputs.seed)
