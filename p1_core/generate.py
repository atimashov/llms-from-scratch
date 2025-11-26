import torch

import yaml
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from datetime import datetime
from termcolor import colored

from p1_core.models import TransformerLM
from p1_core.tokenizers import BPETokenizer
from p1_core.utils import parse_config, load_checkpoint


seed = 123
torch.manual_seed(seed)

class Generator:
    """
    TODO:
        - top-p sampling
        - batched generation
        - parallel decoding
    """
    def __init__(self, config: dict, benchmark = False):
        model_params, model_path, tokenizer_params, vocab_merges_path = parse_config(config, mode = "generate")
        self.device = model_params["device"]
        self.tokenizer = self._init_tokenizer(tokenizer_params, vocab_merges_path)
        self.eof_token = self.tokenizer.eof_token
        self.model = self._init_model(model_params, model_path)
        self.cntx = self.model.context_length
        self.n_steps = config.get("max_num_tokens", 500)
        self.kv_cache = model_params["kv_cache"]
        if benchmark: 
            self.benchmark_gen = []

    def _init_model(self, model_params: dict, model_path: str):
        model = TransformerLM(**model_params)
        load_checkpoint(model_path, model, device=model_params["device"], remap=True)
        model.eval()
        return model
    
    def _init_tokenizer(self, tokenizer_params: dict, vocab_merges_path: Path):
        tokenizer = BPETokenizer(**tokenizer_params)
        vocab_path, merges_path = vocab_merges_path / "vocab.pkl", vocab_merges_path / "merges.pkl"
        tokenizer.from_files(vocab_path, merges_path)
        return tokenizer        

    def generate_next(self, tokens, tau: float = 1.0, topk: int | None = None):
        pred = self.model(tokens, prob = True, tau = tau)[0, -1,:] # NOTE: to make it working for batches, start from changing indexing
        # sample
        if topk is None:
            topk = pred.shape[0]
        probs, indices = torch.topk(pred, topk)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return indices[sampled_idx]

    def generate_tokens(self, tokens, curr_pos: int, tau: float, topk: int | None = None):
        curr_pred = None
        step = 0 
        tokens_pred = []
        circled = False

        while curr_pred != self.eof_token and step < self.n_steps:
            if hasattr(self, "benchmark_gen"):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = perf_counter()
            if circled:
                idx = (torch.arange(self.cntx, device=tokens.device) + curr_pos) % self.cntx
                window = tokens[:, idx]
            else:
                window = tokens[:,:curr_pos]

            curr_pred = self.generate_next(window, tau, topk)
            if self.kv_cache:
                tokens = curr_pred.unsqueeze(1)
            else: 
                tokens[:, curr_pos] = curr_pred
                curr_pos = (curr_pos + 1) % self.cntx
                if curr_pos == 0:
                    circled = True
            
            curr_idx = curr_pred.item()
            if curr_idx != self.eof_token:
                tokens_pred.append(curr_idx)
            step += 1

            if hasattr(self, "benchmark_gen"):
                self.benchmark_gen.append(1000.0 * (perf_counter() - t))

        return tokens_pred

    def generate(self, prompt: str, tau: float = 1.0, topk: int | None = None):
        # encode
        assert len(prompt) > 0, "We expect some prompt, but you entered nothing."
        tokens_list = self.tokenizer.encode_prev(prompt)
        
        # truncate prompt & convert it to torch
        tokens = torch.as_tensor(tokens_list[-self.cntx:], device = self.device, dtype = torch.long)

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        
        B, S = tokens.shape
        if tokens.shape[1] < self.cntx and not self.kv_cache:
            pad = torch.zeros((B, self.cntx - S), device=tokens.device, dtype=torch.long)
            tokens = torch.cat([tokens, pad], dim=1)
        # generate
        tokens_pred = self.generate_tokens(tokens, S % self.cntx, tau, topk)
        # print([(i, t) for i, t in enumerate(tokens_pred)])
        # decode TODO: modify to decode on the fly
        return self.tokenizer.decode(tokens_pred), len(tokens_pred)

if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    curr_time = datetime.now().time()
    # read config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='p1_core/configs/gen.yaml', help='config file')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Prompt to generate text')
    parser.add_argument('--tau', type=float, default=0.7, help='Temperature')
    parser.add_argument('--topk', type=int, default=100, help='Top K')
    parser.add_argument('--benchmark', type=bool, default=False)
    
    inputs = parser.parse_args()
    with open(inputs.config, 'r') as stream:
        config = yaml.safe_load(stream)    
    
    print(colored(f"⏱️ Generation started at {curr_time.strftime("%H:%M:%S")}.", 'red', attrs=["bold"]))
    attn_params = config['model']['attention']
    print(
        f"{colored('Attention type: ', 'blue', attrs=["bold"])}{attn_params['type'].upper()} | "
        f"{colored('Heads: ', 'blue', attrs=["bold"])}{attn_params['num_heads']} | "
        f"{colored('Heads KV: ', 'blue', attrs=["bold"])}{attn_params['num_heads_kv']} |"
        f"{colored('d_latent: ', 'blue', attrs=["bold"])}{attn_params['d_latent']} | "
        f"{colored('Context length: ', 'blue', attrs=["bold"])}{config['model']['context_length']} | "
        f"{colored('KV-cache: ', 'blue', attrs=["bold"])}{config['kv_cache']}"
    )
    
    gen = Generator(config, benchmark = inputs.benchmark)
    t = perf_counter()
    output, num_tokens = gen.generate(inputs.prompt, tau = inputs.tau, topk=inputs.topk)
    t = perf_counter() - t
    print(f"{colored('Generated', 'blue', attrs=["bold"])} {num_tokens} {colored('tokens', 'blue', attrs=["bold"])}")
    print(colored("*" * 100, 'blue', attrs=["bold"]))
    print(f"{colored('Prompt:\n', 'blue', attrs=["bold"])}{inputs.prompt}")
    print(colored("-" * 100, 'blue', attrs=["bold"]))
    print(f"{colored('Generation:\n', 'blue', attrs=["bold"])}{output}")
    print(colored("*" * 100, 'blue', attrs=["bold"]))
    print(f"{colored('⏱️ Generation time:', 'red', attrs=["bold"])}{t:.2f}s")
    if inputs.benchmark:
        mean_time = sum(gen.benchmark_gen[5:]) / len(gen.benchmark_gen[5:])
        print(f"Average generation (after 5 steps): {mean_time:.2f} ms.")
        print(f"First 10 steps: {[round(x, 2) for x in gen.benchmark_gen[:10]]}")
        print()
        
