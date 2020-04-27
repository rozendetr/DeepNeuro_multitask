import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

import os
import argparse

from models import *
from utils import progress_bar
from transform import trainloader, testloader
from models.utils import load_net, save_net
from loss import *


def test(loader):
    net.eval()
    loader_keys = loader.keys()
    total = {key: 0 for key in loader_keys}
    correct = {key: 0 for key in loader_keys}
    count_batches = min({key: len(loader[key]) for key in loader_keys}.values())
    iter_loader = {key: iter(loader[key]) for key in loader_keys}

    with torch.no_grad():
        for batch_idx in range(count_batches):
            batch = {key: next(iter_loader[key]) for key in loader_keys}
            input = {key: batch[key][0].to(device) for key in loader_keys}
            target = {key: batch[key][1].to(device) for key in loader_keys}

            output = {}
            if 'cifar10' in loader_keys:
                output['cifar10'], _ = net(input['cifar10'])
            if 'fashionmnist' in loader_keys:
                _, output['fashionmnist'] = net(input['fashionmnist'])

            predicted = {}
            for key in loader_keys:
                total[key] += target[key].size(0)
                _, predicted[key] = output[key].max(1)
                correct[key] += predicted[key].eq(target[key]).sum().item()

            acc = {key: 100. * correct[key] / total[key] for key in loader_keys}
            str_acc = " ".join(F"acc_{key}: {acc[key]:.3f}" for key in loader_keys)

            progress_bar(batch_idx, count_batches, str_acc)
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training multitask neuronet')
    parser.add_argument('--ckpt', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--heads', type=str, default='both', help="trained heads: both, "
                                                                  "h1(cifar10), or h2(fashion_mnist)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('select device: ', device)

    print('==> Building model..')
    net = ResNet_2head()
    net = net.to(device)

    if args.ckpt:
        print('==> Resuming from checkpoint..')
        if not os.path.isfile(args.ckpt):
            print('Error: no checkpoint file found!')
        else:
            state_dict, best_accs, start_epoch = load_net(args.ckpt)
            net.load_state_dict(state_dict)

    if args.heads == "h1":
        testloader = {'cifar10': testloader['cifar10']}
        name_save_file = "chpt_resnet34_cifar10.pth"
    elif args.heads == "h2":
        testloader = {'fashionmnist': testloader['fashionmnist']}
        name_save_file = "chpt_resnet34_fashionmnist.pth"

    test_accs = []

    test_acc = test(testloader)
    print('accuracity: ', list(test_acc.values()))


