import torch
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import os
from time import time
import platform

from models import *
from utils import *
from poc_enc import *
from norms import *
from bpe_tokenizer import *



seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


def train_loop(steps, model, optimizer, loss_fn, tokens_data, batch_size, context_length, device, dtype):
    model.train()  # put model to training mode TODO: should I do it every step?
    loop = tqdm(range(steps), leave = True)
    for n_step in loop:
        # TODO: update optmizer
        tokens_curr, tokens_next = data_loading(tokens_data, batch_size, context_length, device)
        logits = model(token_curr)
        loss = loss_fn(logits, tokens_next)

        # Zero out all of the gradients for the variables which the optimizer will update.
        optimizer.zero_grad()
        # Backwards pass and computing gradients
        loss.backward()
        optimizer.step()
        
        # TODO: serialize
        # TODO: do logging
        
        # update progress bar
        loop.set_postfix(loss=loss.item())




def main(config):
    # model
    if config["gpu"] is None:
        device = "mps" if platform.system() == 'Darwin' else: "cpu"
    else:
        device = "cuda:{}".format(config["gpu"])
    model_params = {
        "d_model": config["model"]["d_model"],
        "d_ff": config["model"]["d_ff"],
        "num_heads": config["model"]["num_heads"],
        "num_layers": config["model"]["num_layers"],
        "rope_theta": config["model"]["rope_theta"],
        "context_length": config["model"]["context_length"],
        "vocab_size": config["model"]["vocab_size"],
        "device": torch.device(device),
        "dtype": torch.float32 # TODO: improve later
    }
    model = TransformerLM(**model_params)
    # optimizer
    opt_params = {
        "params": model.parameters(),
        "lr": config["optimizer"]["lr"],
        "betas": (config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
        "weight_decay": config["weight_decay"],
        "eps": config["epsilon"],
        decoupled = True
    }
    opt = Adam(**opt_params)
    # scheduler:
    #     name: cosine
    #     lr_min: 1e-6
    #     warmup_iters: 0.1
    #     cosine_cycle_iters: 1.0
    
    # loss 
    loss_fn = cross_entropy

    # data (tokenize)
    # TODO: tokenize 
    if config["dataset_path"]["tokenized"] is None:
        bpe_params = {
            "input_path": config["dataset_path"]["raw"], 
            "vocab_size": config["model"]["vocab_size"], 
            "special_tokens": ["<|endoftext|>"]
        }
        bpe = BPETokenizer(**bpe_params)
        # train BPE tokenizer
        bpe.train()
        # encode text (TODO: test "encode_iterable")
        token_ids bpe.encode(self, text: str, num_processes: int = 24)
        # TODO :save it to use as np.memmap

    tokenized_data = None
    
    # train
    batch_size = config["train"]["batch_size"]
    context_length = config["model"]["context_length"]
    steps = int(config["train"]["total_tokens_processed"] / (batch_size * context_length))


    device = torch.device(inputs.device)
    dtype = inputs.device
    load_checkpoint_src = inputs.checkpoint_src
    


    optimizer = optim.Adam(
        model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    loader_params = BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DATASET_DIR, CSV_TRAIN, CSV_VAL
    loader = get_dataloader(loader_params)

    # create folder to save models
    if SAVE_MODEL:
        if not os.path.exists('{}/{}'.format(SAVE_MODEL_DIR, MODEL)):
            os.makedirs('{}/{}'.format(SAVE_MODEL_DIR, MODEL))
    losses, accuracies = {'train': [], 'validate': []}, {'train': [], 'validate': []}

    for epoch in range(start_epoch, EPOCHS + start_epoch):
        t = time()
        if (epoch + 1) % DECAY_EPOCHS == 0:
            L_RATE *= (1 - DECAY_RATE)
            optimizer = optim.Adam(model.parameters(), lr=L_RATE, weight_decay=WEIGHT_DECAY)

        # print epoch number
        print_report(part='start', epoch=epoch)
        # train loop
        train_epoch(loader['train'], model, optimizer, device, loss_fn)

        # print metrics
        pred_bb, target_bb = get_bboxes(
            loader['train'], model, iou_threshold=0.5, threshold=0.4
        )
        train_map = mean_average_precision(
            pred_bb, target_bb, iou_threshold = 0.5, box_format = 'midpoint'
        )

        v_pred_bb, v_target_bb = get_bboxes(
            loader['val'], model, iou_threshold=0.5, threshold=0.4
        )
        val_map = mean_average_precision(
            v_pred_bb, v_target_bb, iou_threshold=0.5, box_format='midpoint'
        )

        metrics = -1, -1, train_map, val_map
        print_report(part='accuracy', metrics=metrics)
        # collect metrics
        # losses['train'].append(train_loss)
        # losses['validate'].append(val_loss)
        # accuracies['train'].append(train_acc)
        # accuracies['validate'].append(val_acc)

        # save models
        # if SAVE_MODEL:
        #     save_checkpoint(model=model, cfg=cfg, epoch=epoch, loss=round(val_loss, 3))

        # print time
        print_report(part='end', t=int(time() - t))


if __name__ == '__main__':
    seed = 123
    torch.manual_seed(seed)

    # read config
    parser = ArgumentParser()
    parser.add_argument('--gpu-number', type=str, default='0', help='GPU number: 0 or 1')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')

    inputs = parser.parse_args()
    print(inputs)
    gpu_n = inputs.gpu_number
    cfg_path = inputs.config
    train_loop(cfg_path, gpu_n='0')
