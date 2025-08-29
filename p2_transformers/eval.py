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

from models import TransformerLM
from utils import parse_config, eval, load_checkpoint, cross_entropy, count_parameters, get_short_gpu_name


def _init_model(model_params, model_path):
    model = TransformerLM(**model_params)
    load_checkpoint(model_path, model, None, model_params["device"])
    model.eval()
    return model

def main(config):
    model_params, model_path, tokens_path = parse_config(config, "eval")

    model = _init_model(model_params, model_path)
    tokens = np.load(tokens_path, mmap_mode='r')
    loss_fn = cross_entropy

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
    final_valid_loss = eval(tokens, model, loss_fn, context_length, batch_size, num_samples, device)

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config_eval.yaml', help='config file')

    inputs = parser.parse_args()
    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)
    main(config)
