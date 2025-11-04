import os
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    # Init process grup: 'gloo' is used for CPU; 'nccl' - for machines with GPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class DDP(nn.Module):
    """Works for 1 node"""
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None = None): 
        # Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks.
        super().__init__()
        assert dist.is_initialized(), "init_process_group must be called before DDP wrapper."
        self.rank = dist.get_rank(dist.group.WORLD)

        self.bucket_size_mb = bucket_size_mb

        self.module = module
        if torch.cuda.is_available():
            self.module.to(f"cuda:{self.rank}")

        # Broadcast parameters and buffers from rank "0"
        self.broadcast_module(self.module, src = 0)

    def forward(self, *inputs, **kwargs): 
        # Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
        return self.module(*inputs, **kwargs)
    
    @torch.no_grad()
    def finish_gradient_synchronization(self, mode: str = "flat"):
        assert mode in {"baseline", "flat", "mix"}, "Wrong mode"
        ws = dist.get_world_size()
        
        if mode == "baseline":
            handles = []
            for p in reversed(self.module.parameters()):
                if p.grad is None:
                    g = torch.zeros_like(p, memory_format=torch.preserve_format)
                    p.grad = g
                p.grad.div_(ws) 
                handles.append(dist.all_reduce(p.grad, async_op = True, op = dist.ReduceOp.SUM))
        
            # ensure that all async operations were queued
            for h in handles:
                h.wait()
            handles.clear()

        elif mode == "flat":
            grads = [p.grad if p.grad is not None else torch.zeros_like(p, memory_format=torch.preserve_format) for p in self.module.parameters()]

            flat_g = _flatten_dense_tensors(grads)
            flat_g.div_(ws)

            handle = dist.all_reduce(flat_g, async_op = True, op = dist.ReduceOp.SUM)
            handle.wait()
            
            for p, ufg in zip(grads, _unflatten_dense_tensors(flat_g, grads)):
                p.copy_(ufg)

        elif mode == "mix":
            grads, buffer = [], []
            num_bytes, bucket_bytes_cap = 0, int(self.bucket_size_mb * 1024 * 1024)

            for p in reversed(list(self.module.parameters())):
                if p.grad is None:
                    p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)

                g_bytes = p.grad.numel() * p.grad.element_size()
                if num_bytes + g_bytes > bucket_bytes_cap and num_bytes > 0:
                    # flush current bucket
                    flat_g = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flat_g, async_op = True, op = dist.ReduceOp.SUM)
                    buffer.append((flat_g, grads, handle))
                    grads.clear()
                    num_bytes = 0

                # append new bucket
                grads.append(p.grad)
                num_bytes += g_bytes

            # flush tail bucket
            if num_bytes > 0:
                flat_g = _flatten_dense_tensors(grads)
                handle = dist.all_reduce(flat_g, async_op = True, op = dist.ReduceOp.SUM)
                buffer.append((flat_g, grads, handle))
                grads.clear()

            # finalize all-reduce
            for flat_g, grads, h in buffer:
                h.wait()
                flat_g.div_(ws)                    
                for g_dst, g_uf in zip(grads, _unflatten_dense_tensors(flat_g, grads)):
                    g_dst.copy_(g_uf)
            buffer.clear
            
    @torch.no_grad()
    def broadcast_module(self, model, src = 0):
        # ----- broadcast parameters -----
        params = [p.detach() for p in model.parameters()]
        if params:                                                  
            flat_p = _flatten_dense_tensors(params)
            dist.broadcast(flat_p, src=src)
            for p, ufp in zip(model.parameters(), _unflatten_dense_tensors(flat_p, params)):
                p.copy_(ufp)

        # ------ broadcast buffers ------
        buffs = [p.detach() for p in model.buffers()]
        if buffs:
            flat_b = _flatten_dense_tensors(buffs)
            dist.broadcast(flat_b, src = src)
            for b, ufb in zip(model.buffers(), _unflatten_dense_tensors(flat_b, buffs)):
                b.data.copy_(ufb)
