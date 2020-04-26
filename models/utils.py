
import torch
import os

def load_net(net, path_load):
    """
    Load net from file ckpt
    :param net:
    :param path_load:
    :return:
    """
    print('==> Loaded net..')
    checkpoint = torch.load(path_load)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net


def save_net(net, path_save, acc_mean, epoch):
    """
    Save met to file ckpt
    :param net:
    :param path_save:
    :param acc_mean:
    :param epoch:
    :return:
    """
    state = {'net': net.state_dict(),
             'acc': acc_mean,
             'epoch': epoch}
    torch.save(state, path_save)


