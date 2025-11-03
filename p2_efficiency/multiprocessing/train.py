import os
from tqdm import tqdm
from termcolor import colored
import torch
from torch import nn
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp


class ToyModel(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
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
    torch.cuda.set_device(rank)
    # Init process grup: 'gloo' is used for CPU; 'nccl' - for machines with GPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train_parallel(rank, world_size, batch_size, d_in, d_hidden, d_out): # , dataset, loss_fn, opt, x, y, steps):
    setup(rank, world_size)

    # create the same model with random weights
    model = ToyModel(d_in, d_hidden, d_out)
    model.to("cuda")

    # Broadcast parameters and buffers from rank "0"
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        # Broadcast buffers (e.g., BatchNorm running stats)
        for b in model.buffers():
            dist.broadcast(b.data, src=0)

    # init optimizer
    opt = Adam(model.parameters())

    # init loss function
    loss_fn = nn.CrossEntropyLoss()

    # dataset
    x = torch.randn(batch_size, d_in)
    y = torch.randint(0, d_out, (batch_size, ))
    x, y = x.to("cuda"), y.to("cuda")
    
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
    batch_size: int = 128
    d_in: int = 64
    d_hidden: int = 128
    d_out: int = 16


    world_size = 2
    
    mp.spawn(fn=train_parallel, args=(world_size, batch_size, d_in, d_hidden, d_out, ), nprocs=world_size, join=True)