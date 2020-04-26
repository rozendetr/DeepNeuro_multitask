import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse

from models import *
from utils import progress_bar
from transform import trainloader, testloader
from models.utils import load_net, save_net
from loss import *


def train(epoch, loader, optimizer, total_loss, freeze_core=False):
    net.train()
    if freeze_core:
        net.freeze_shared()
    else:
        net.unfreeze_shared()

    loader_keys = loader.keys()
    total = {key: 0 for key in loader_keys}
    correct = {key: 0 for key in loader_keys}
    count_batches = min({key: len(loader[key]) for key in loader_keys}.values())
    iter_loader = {key: iter(loader[key]) for key in loader_keys}

    # total_loss.eval_lambda_weight(epoch)

    for batch_idx in range(count_batches):
        batch = {key: next(iter_loader[key]) for key in loader_keys}
        input = {key: batch[key][0].to(device) for key in loader_keys}
        target = {key: batch[key][1].to(device) for key in loader_keys}

        optimizer.zero_grad()
        output = {}
        output['cifar10'], _ = net(input['cifar10'])
        _, output['fashionmnist'] = net(input['fashionmnist'])

        losses = {key: criterion(output[key], target[key]) for key in loader_keys}

        loss = total_loss.eval_trainloss(losses, epoch, count_batches)
        loss.backward()
        optimizer.step()

        predicted = {}
        for key in loader_keys:
            total[key] += target[key].size(0)
            _, predicted[key] = output[key].max(1)
            correct[key] += predicted[key].eq(target[key]).sum().item()

        acc = {key: 100.*correct[key]/total[key] for key in loader_keys}
        str_acc = " ".join([F"acc_{key}: {acc[key]:.3f}" for key in loader_keys])

        progress_bar(batch_idx, count_batches, str_acc)
    return acc


def test(epoch, loader, total_loss):
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
            output['cifar10'], _ = net(input['cifar10'])
            _, output['fashionmnist'] = net(input['fashionmnist'])

            losses = {key: criterion(output[key], target[key]) for key in loader_keys}

            total_loss.eval_validloss(losses, epoch, count_batches)
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
    # parser.add_argument('--resume', '-r', action='store_true',
    #                     help='resume from checkpoint')
    parser.add_argument('--ckpt', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--out', type=str, default='checkpoint', help='output folder')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('select device: ', device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    count_epoch = 50  # count epoch from start_epoch to end of train

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
            best_acc = sum(best_accs.values()) / len(best_accs.values())
            start_epoch += 1
            print("loaded acc", best_accs)
            
    criterion = nn.CrossEntropyLoss()
    l_rate = 0.1
    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9, weight_decay=5e-4)
    # total_loss = DWALoss(len(trainloader.keys()), count_epoch)
    total_loss = SimpleLoss(len(trainloader.keys()), count_epoch, device)

    train_accs = []
    test_accs = []

    for epoch in range(start_epoch, start_epoch + count_epoch):
        print(F"\nEpoch: {epoch:d}")
        train_acc = train(epoch, trainloader, optimizer, total_loss)
        train_accs.append(train_acc)
        test_acc = test(epoch, testloader, total_loss)
        test_accs.append(test_acc)

        acc_mean = sum(test_acc.values()) / len(test_acc.values())
        if acc_mean > best_acc:
            print('Saving..')
            print(F"acc_mean: {acc_mean}, best_acc: {best_acc}")
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_net('./checkpoint/chpt_resnet_2h.pth', net.state_dict(), test_accs[epoch], epoch)
            best_acc = acc_mean

        print("train loss", total_loss.get_train_avgloss(epoch))
        print("validate loss", total_loss.get_valid_avgloss(epoch))
        print("train acc", train_accs[epoch])
        print("validate acc", test_accs[epoch])

