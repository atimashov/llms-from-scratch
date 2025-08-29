import torch

import yaml
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from datetime import datetime
import numpy as np

from models import TransformerLM
from tokenizers import BPETokenizer
from utils import parse_config, load_checkpoint


seed = 123
torch.manual_seed(seed)

class Generator:
    """
    TODO:
        - memory preallocation for prompt
        - KV Cache
        - top-p sampling
        - AMP/FP16
        - batched generation
        - parallel decoding
    """
    def __init__(self, config: dict):
        model_params, model_path, tokenizer_params, vocab_merges_path = parse_config(config, mode = "generate")
        self.device = model_params["device"]
        self.tokenizer = self._init_tokenizer(tokenizer_params, vocab_merges_path)
        self.eof_token = self.tokenizer.eof_token
        self.model = self._init_model(model_params, model_path)
        self.n_steps = config.get("max_num_tokens", float('inf'))

    def _init_model(self, model_params: dict, model_path: str):
        model = TransformerLM(**model_params)
        load_checkpoint(model_path, model, None, model_params["device"])
        model.eval()
        return model
    
    def _init_tokenizer(self, tokenizer_params: dict, vocab_merges_path: Path):
        tokenizer = BPETokenizer(**tokenizer_params)
        vocab_path, merges_path = vocab_merges_path / "vocab.pkl", vocab_merges_path / "merges.pkl"
        tokenizer.from_files(vocab_path, merges_path)
        return tokenizer        

    def generate_next(self, tokens, tau: float = 1.0, topk: int | None = None):
        pred = self.model(tokens, prob = True, tau = tau)[0, -1] # 
        # sample
        if topk is None:
            topk = pred.shape[0]
        values, indices = torch.topk(pred, topk)
        probs = values / values.sum()
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return indices[sampled_idx].item()

    def generate_tokens(self, tokens, tau: float, topk: int | None = None):
        curr_pred = None
        step = 0 
        tokens_pred = []

        while curr_pred != self.eof_token and step < self.n_steps:
            curr_pred = self.generate_next(tokens, tau, topk)
            add = torch.tensor([[curr_pred]]).to(device = self.device, dtype = torch.long)
            tokens = torch.cat((tokens, add), dim = 1) # TODO: probably can do more efficiently, for example pre-allocate memory
            step += 1
            if curr_pred != self.eof_token:
                tokens_pred.append(curr_pred)
        return tokens_pred

    def generate(self, prompt: str, tau: float = 1.0, topk: int | None = None):
        # encode
        tokens_list = self.tokenizer.encode(prompt)
        tokens = torch.as_tensor(tokens_list, device = self.device, dtype = torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        # generate
        tokens_pred = self.generate_tokens(tokens, tau, topk)
        # decode TODO: modify to decode on the fly
        return self.tokenizer.decode(tokens_pred)

            

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config_gen.yaml', help='config file')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Prompt to generate text')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature')
    parser.add_argument('--topk', type=int, default=100, help='Top K')
    
    inputs = parser.parse_args()
    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)    
    gen = Generator(config)
    t = perf_counter()
    output = gen.generate(inputs.prompt, tau = inputs.tau, topk=inputs.topk)
    print(f"⏱️ Generation time={perf_counter() - t:.2f}s")
    print("*" * 100)
    print(inputs.prompt)
    print("-" * 100)
    print(output)