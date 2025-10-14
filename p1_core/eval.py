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
from pathlib import Path

from models import TransformerLM
from utils import parse_config, eval, load_checkpoint, cross_entropy, count_parameters, get_short_gpu_name
from termcolor import colored


def _init_model(model_params, model_path):
    
    # Initialize and load model
    model = TransformerLM(**model_params)
    if config["train"]["compile"]:
        model = torch.compile(model)
    a, b, c = load_checkpoint(model_path, model, None, model_params["device"])
    print("*** ", b, c, " ***")
    model.eval()
    return model

def main(config):
    model_params, model_path, tokens_path = parse_config(config, "eval")
    print(model_path)
    print(tokens_path)
    model = _init_model(model_params, model_path)

    tokens = np.load(tokens_path, mmap_mode='r')
    print("tokens: ", tokens.shape)
    def loss_fn(logits: torch.Tensor, target: torch.Tensor, z_alpha:float = 0.0): #config["train"]["z_alpha"]):
        return cross_entropy(logits, target, float(z_alpha))

    context_length = model_params["context_length"]
    device = model_params["device"]
    batch_size = config["batch_size"]
    max_available, num_samples = tokens.shape[0] - context_length, config["data"]["num_samples"]
    num_samples = max_available if num_samples < 0 else min(num_samples, max_available)

    print()
    curr_time = datetime.now().time()
    print(
        f"Validation started at {curr_time.strftime("%H:%M:%S")}: Num of samples={num_samples:,} | "
        f"Num of params={count_parameters(model):,} | "
        f"Context={model_params['context_length']} | "
        f"Vocab_size={model_params['vocab_size']:,} | "
        f"Device={get_short_gpu_name(config["device"])}"
    )
    final_valid_loss = eval(-1, tokens, model, loss_fn, context_length, batch_size, num_samples, device, use_amp = True)
    print(f"Final validation loss for {num_samples} samples = {final_valid_loss:.3f}")

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    # parser = ArgumentParser()
    # parser.add_argument('--config', type=str, default='configs/eval.yaml', help='config file')

    # inputs = parser.parse_args()
    # with open(inputs.config, 'r') as stream:
    #     config = yaml.safe_load(stream)
    
    path_names = [
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250921_200000_sunday",
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250922_013251_sunday",
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250922_095334_sunday",
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250921_195921_sunday",
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250922_025413_sunday",
        "~/ai_projects/llms-from-scratch/p1-core/weights/OpenWebText/wandb/20250922_113500_sunday"
    ]
    # TODO: check on random data
    for path_name in path_names:
        with open(Path(path_name).expanduser() / "config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        config["model"]["load_prefix"] = path_name
        config["model"]["load_name"] = "ckpt_best.pt"
        config["device"] = 1
        config["data"] = {"path": '~/ai_projects/data/OpenWebText/tokenized/valid.npy'}
        config["batch_size"] = 64
        
        for n in [1280, 32000, 320_000]:
            print()
            print(f"{path_name.split("/")[-1]}: {n if n > 0 else 'all'}")
            config["data"]["num_samples"] = n
  
            try:
                t_start = perf_counter()
                main(config)
                elapsed = perf_counter() - t_start
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                print(colored(f"⏱️ Took {minutes} min {seconds} sec", "green", attrs=["bold"]))
            except:
                print(colored("⚠️  Bad state", "red", attrs=["bold"]))
            print()
        print(colored('-' * 150, 'cyan', attrs=["bold"]))
