import os
import torch
from torch import nn
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp

"""
NOTE:
When running GPU dostributed job, we need to ensure that 
different ranks use different GPUS:
- torch.cuda.set_device(rank) -> tensor.to('cuda')
- device = f"cuda:{rank}"
"""

class ToyModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        self.W1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(d_hidden, d_out)
    
    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        return x

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


def train_parallel(rank, world_size, dataset, model, loss_fn, opt, x, y, steps):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # dataset
    X, Y = dataset
    MN = X.shape[0]
    M = MN // world_size
    x = X[rank * M: (rank + 1) * M]
    y = Y[rank * M: (rank + 1) * M]
    x, y = x.to(device), y.to(device)
    
    loop = tqdm(
        range(1000),
        leave = True,
        desc = colored("Training", 'blue', attrs=["bold"])
    )
    for i in loop:
        opt.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()

        # all reduce
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, async_op=False)
                p.grad /= world_size

        opt.step()

    
   


if __name__ == "__main__":
    batch_size: int = 128, d_in: int: 64, d_hidden: int = 128, d_out: int = 16


    world_size = 4
    # Create 'world_size' processes; each of them runs function fn (in out case 'distributed demo') with arguments (rank, *args)
    

    # init model
    model = ToyModel(d_in, d_hidden, d_out) # TODO: and this?

    # init optimizer
    opt = Adam(model.parameters())

    # init data
    x = torch.randn(batch_size, d_in) # TODO: how are we going to process it here?
    y = torch.randint(0, d_out, (batch_size, )) # TODO: how are we going to process it here?



    loss_fn = nn.CrossEntropyLoss()
    
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)