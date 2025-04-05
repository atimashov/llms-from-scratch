import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from time import time
from argparse import ArgumentParser

from models import SkipGram
from datasets import StanfordSentiment
from losses import Word2VecLoss
from utils import print_start, print_end, save_checkpoint



def train_epoch(loader, model, optimizer, loss_fn, device = 'cpu'):
    loop = tqdm(loader, leave = True)

    for center_ids, context_ids in loop:
        model.train()
        # center_ids, context_ids = center_ids.to(device), context_ids.to(device)
        cos_positive, cos_negative = model(center_ids, context_ids)
        loss = loss_fn(cos_positive, cos_negative)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item()) # NOTE: Add time
    return loss.item()


def train_loop(batch_size, epochs, lr):
    data = StanfordSentiment()
    loader = DataLoader(
        data, batch_size = batch_size, shuffle = True, num_workers = 6, drop_last=True, pin_memory = True
    )
    model = SkipGram(vocab_size=len(data._tokens), neg_sample=5, freq=data._tokenfreq_list)
    save_checkpoint(model=model, epoch='pre', loss='NA', lr = lr, batch_size = batch_size)
    loss_fn = Word2VecLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0)
    
    # start training
    for epoch in range(epochs):
        t = time()
        print_start(epoch)
        loss = train_epoch(loader, model, optimizer, loss_fn)
        print_end(int(time() - t))
        # save model
        save_checkpoint(model=model, epoch=epoch, loss=round(loss, 3), lr = lr, batch_size = batch_size)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type = int, default = 4, help = 'Size of the batch')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Number of epochs to train')
    inputs = parser.parse_args()
    train_loop(inputs.batch_size, inputs.epochs, inputs.lr)