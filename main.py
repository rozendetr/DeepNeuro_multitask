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

    total_loss.eval_lambda_weight(epoch)

    for batch_idx in range(count_batches):
        batch = {key: next(iter_loader[key]) for key in loader_keys}
        input = {key: batch[key][0].to(device) for key in loader_keys}
        target = {key: batch[key][1].to(device) for key in loader_keys}

        optimizer.zero_grad()
        output = {}
        if 'cifar10' in loader_keys:
            output['cifar10'], _ = net(input['cifar10'])
        if 'fashionmnist' in loader_keys:
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
            if 'cifar10' in loader_keys:
                output['cifar10'], _ = net(input['cifar10'])
            if 'fashionmnist' in loader_keys:
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
    parser.add_argument('--ckpt', type=str, default='', help='resume from checkpoint')
    parser.add_argument('--out', type=str, default='checkpoint', help='output folder')
    parser.add_argument('--freeze_core', '-r', action='store_true', help='freeze shared core')
    parser.add_argument('--heads', type=str, default='both', help="trained heads: both, "
                                                                  "h1(cifar10), or h2(fashion_mnist)")
    parser.add_argument('--dwa', action='store_true', help='use Dynamic Weight Average')
    # Dynamic Weight Average - method adaptive weighting method, named Dynamic Weight Aver-age (DWA)
    # from article https://arxiv.org/pdf/1803.10704.pdf
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('select device: ', device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    count_epoch = 200  # count epoch from start_epoch to end of train

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
    start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    l_rate = 0.1
    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if args.heads == "h1":
        trainloader = {'cifar10': trainloader['cifar10']}
        testloader = {'cifar10': testloader['cifar10']}
        print(F"Train only head_1: {trainloader.keys()}")
    elif args.heads == "h2":
        trainloader = {'fashionmnist': trainloader['fashionmnist']}
        testloader = {'fashionmnist': testloader['fashionmnist']}
        print(F"Train only head_2: {trainloader.keys()}")

    if not args.dwa:
        print("Use SimpleLoss")
        total_loss = SimpleLoss(len(trainloader.keys()), count_epoch, device)
    else:
        print("Use DWALoss")
        total_loss = DWALoss(len(trainloader.keys()), count_epoch)

    train_accs = []
    test_accs = []

    for epoch in range(start_epoch, start_epoch + count_epoch):
        print(F"\nEpoch: {epoch:d}")
        if args.freeze_core:
            print("Freezed share core")
            train_acc = train(epoch, trainloader, optimizer, total_loss, freeze_core=True)
        else:
            train_acc = train(epoch, trainloader, optimizer, total_loss, freeze_core=False)
        train_accs.append(list(train_acc.values()))
        test_acc = test(epoch, testloader, total_loss)
        test_accs.append(list(test_acc.values()))
        print(F"\nLearning_rate: {scheduler.get_lr()[0]}")
        scheduler.step()
        acc_mean = sum(test_acc.values()) / len(test_acc.values())
        if acc_mean > best_acc:
            print('Saving..')
            print(F"acc_mean: {acc_mean}, best_acc: {best_acc}")
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_net('./checkpoint/chpt_resnet_2h.pth', net.state_dict(), test_acc, epoch)
            best_acc = acc_mean

        print("train loss", total_loss.get_train_avgloss(epoch))
        print("validate loss", total_loss.get_valid_avgloss(epoch))
        print("train acc", train_acc)
        print("validate acc", test_acc)

        pd.DataFrame(train_accs, columns=trainloader.keys()).to_csv('train_accs.csv')
        pd.DataFrame(test_accs, columns=trainloader.keys()).to_csv('test_accs.csv')
        pd.DataFrame(data=total_loss.get_train_avglosses(), columns=trainloader.keys()).to_csv('train_losses.csv')
        pd.DataFrame(data=total_loss.get_valid_avglosses(), columns=trainloader.keys()).to_csv('test_losses.csv')

    pd.DataFrame(train_accs, columns=trainloader.keys()).to_csv('train_accs.csv')
    pd.DataFrame(test_acc, columns=trainloader.keys()).to_csv('test_acc.csv')
    pd.DataFrame(data=total_loss.get_train_avglosses(), columns=trainloader.keys()).to_csv('train_losses.csv')
    pd.DataFrame(data=total_loss.get_valid_avglosses(), columns=trainloader.keys()).to_csv('test_losses.csv')


