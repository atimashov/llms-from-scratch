import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

"""
NOTE:
When running GPU dostributed job, we need to ensure that 
different ranks use different GPUS:
- torch.cuda.set_device(rank) -> tensor.to('cuda')
- device = f"cuda:{rank}"
"""

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Init process grup: 'gloo' is used for CPU; 'nccl' - for machines with GPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")


if __name__ == "__main__":
    world_size = 4
    # Create 'world_size' processes; each of them runs function fn (in out case 'distributed demo') with arguments (rank, *args)
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)