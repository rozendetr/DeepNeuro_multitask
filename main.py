import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse

from models import *
from utils import progress_bar
from transform import trainloader, testloader
from models.utils import *
from loss import *


def train(epoch, loader, optimizer, total_loss, freeze_core=False):
    print('\nEpoch: %d' % epoch)
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
        output['cifar10'], _ = net(input['cifar10'])
        _, output['fashionmnist'] = net(input['fashionmnist'])

        losses = {key: criterion(output[key], target[key]) for key in loader_keys}
        cost = {key: losses[key].item() for key in loader_keys}

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


def test(epoch, loader, total_loss, best_acc):
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

    acc_mean = sum(acc.values())/len(acc.values())

    if acc_mean > best_acc:
        print('Saving..')
        print(F"acc_mean: {acc_mean}, best_acc: {best_acc}")
        state = {'net': net.state_dict(),
                 'acc': acc_mean,
                 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/chpt_resnet_2h.pth')
        best_acc = acc_mean
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training multitask neuronet')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('select device: ', device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    count_epoch = 50  # count epoch from start_epoch to end of train

    print('==> Building model..')
    net = ResNet_2head()
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    l_rate = 0.1
    optimizer = optim.SGD(net.parameters(), lr=l_rate, momentum=0.9, weight_decay=5e-4)
    total_loss = DWALoss(len(trainloader.keys()), count_epoch)

    train_acc = []
    test_acc = []

    for epoch in range(start_epoch, start_epoch + count_epoch):
        train_acc.append(train(epoch, trainloader, optimizer, total_loss))
        test_acc.append(test(epoch, testloader, total_loss, best_acc))
        print("train loss", total_loss.get_train_avgloss(epoch))
        print("validate loss", total_loss.get_valid_avgloss(epoch))
        print("train acc", train_acc[epoch])
        print("validate acc", test_acc[epoch])

