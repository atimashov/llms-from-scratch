from termcolor import colored
import torch
import os

def print_start(epoch):
    print('{} Epoch {}{} {}'.format(' ' * 60, ' ' * (3 - len(str(epoch))), epoch, ' ' * 61))
    print(' ' * 132)


def print_end(t):
    t_min, t_sec = str(t // 60), str(t % 60)
    txt = 'It took {}{} min. {}{} sec.'.format(' ' * (2 - len(t_min)), t_min, ' ' * (2 - len(t_sec)), t_sec)
    print(txt)
    print()
    print(colored('-' * 132, 'cyan'))
    print()


def save_checkpoint(model, epoch, loss, lr, batch_size):
    d = {
        'model': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    lr = str(lr).replace('.', '_')
    if 'weights' not in os.listdir():
        os.mkdir('weights')
    if lr not in os.listdir('weights'):
        os.mkdir('weights/{}'.format(lr))
    torch.save(d, 'weights/{}/word2vec_bs_{}{}_epoch{}{}.pt'.format(lr, " " * (2 - len(str(batch_size))), batch_size, " " * (3 - len(str(epoch))), epoch))