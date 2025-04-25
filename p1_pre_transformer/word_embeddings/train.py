import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from time import perf_counter
from datetime import datetime
from argparse import ArgumentParser
import os

from models import SkipGram, GloVe
from datasets import StanfordSentiment, GloveDataSet
from losses import Word2VecLoss, GloveLoss
from utils import print_start, print_end, save_checkpoint



def train_epoch_skipgram(loader, model, optimizer, loss_fn, device = 'cpu', gpu_batched = True):
    loop = tqdm(loader, leave = True)

    for center_ids, context_ids in loop:
        center_ids, context_ids = center_ids.to(device), context_ids.to(device)
        cos_positive, cos_negative = model(center_ids, context_ids)
        loss = loss_fn(cos_positive, cos_negative)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item()) # NOTE: Add time
    return loss.item()

def train_epoch_glove(loader, model, optimizer, loss_fn, device = 'cpu', gpu_batched = True):
    """
    GPU batched?
    """
    loop = tqdm(loader, leave = True)
    for pairs, cooccurence in loop:
        if device == 'cpu' or not gpu_batched:
            pairs, cooccurence = pairs.to(device), cooccurence.to(device)
        weight, delta = model(pairs, cooccurence)
        loss = loss_fn(weight, delta)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss = loss.item()) # NOTE: Add time
    return loss.item()


def train_loop(use_case, batch_size, epochs, lr, device, num_workers = None, gpu_batched = True):
    assert use_case in {'skipgram', 'glove'}
    print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] Train loop started")
    if use_case == 'skipgram':
        train_epoch = train_epoch_skipgram
        data = StanfordSentiment()
        model = SkipGram(vocab_size=len(data._tokens), neg_sample=5, freq=data._tokenfreq_list, device = device)
        loss_fn = Word2VecLoss()
    else:
        train_epoch = train_epoch_glove
        data = GloveDataSet(files_path = '/content/GloVe_training', chunk_folder_name = 'torch_tensors_50M')
        model = GloVe(vocab_size=len(data._id_to_tokens))
        loss_fn = GloveLoss()
    if num_workers is None:
        num_workers = min(4, os.cpu_count() - 1)
    print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] Creating DataLoader")
    loader = DataLoader(
        data, batch_size = batch_size, shuffle = use_case != 'glove', num_workers = num_workers, drop_last=False, pin_memory = device != 'cpu'
    )
   
    model = model.to(device)
    model.train()
    save_checkpoint(model=model, epoch='pre', loss='NA', lr = lr, batch_size = batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}] Start epoch")
    # start training
    for epoch in range(epochs):
        t = perf_counter()
        print_start(epoch)
        loss = train_epoch(loader, model, optimizer, loss_fn, device = device, gpu_batched = gpu_batched)
        print_end(int(perf_counter() - t))
        # save model
        save_checkpoint(model=model, epoch=epoch, loss=round(loss, 3), lr = lr, batch_size = batch_size)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--use-case", 
        type=str, 
        required=True, 
        choices=["glove", "skipgram"],
        help="Use case to train"
    )
    parser.add_argument('--batch-size', type = int, default = 4, help = 'Size of the batch')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Number of epochs to train')
    parser.add_argument('--device', type = str, default = 'cuda:0', help = 'cuda, cuda:<n>, or cpu')
    inputs = parser.parse_args()
    # TODO: check CPU 
    train_loop(inputs.use_case, inputs.batch_size, inputs.epochs, inputs.lr, inputs.device)