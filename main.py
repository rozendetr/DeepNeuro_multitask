import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse

from models import *
from utils import *
from transform import *
from models.utils import *
from loss import *


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

def train(epoch, loader, freeze_core=False):
    global total_loss
    print('\nEpoch: %d' % epoch)
    net.train()
    net.freeze_shared(freeze_core)
    loader_keys = loader.keys()

    train_loss = 0
    total = {key: 0 for key in loader_keys}
    correct = {key: 0 for key in loader_keys}
    len_loader = {key: len(loader[key]) for key in loader_keys}
    iter_loader = {key: iter(loader[key]) for key in loader_keys}
    min_len_loader = min(len_loader.values())

    total_loss.eval_lambda_weight(epoch)

    for batch_idx in range(min_len_loader):
        batch = {key: next(iter_loader[key]) for key in loader_keys}
        input = {key: batch[key][0].to(device) for key in loader_keys}
        target = {key: batch[key][1].to(device) for key in loader_keys}

        optimizer.zero_grad()
        output = {}
        output['cifar10'], _ = net(input['cifar10'])
        _, output['fashionmnist'] = net(input['fashionmnist'])

        losses = {key: criterion(output[key], target[key]) for key in loader_keys}
        cost = {key: losses[key].item() for key in loader_keys}

        loss = total_loss.eval_trainloss(losses, epoch, min_len_loader)

        # loss = max(losses['cifar10'], losses['fashionmnist'])
        # loss = losses['cifar10'] + losses['fashionmnist']
        # loss = (losses['cifar10'] + losses['fashionmnist'])/2

        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        batch_loss = total_loss.get_train_avgloss(epoch)

        predicted = {}
        for key in loader_keys:
            total[key] += target[key].size(0)
            _, predicted[key] = output[key].max(1)
            correct[key] += predicted[key].eq(target[key]).sum().item()

        acc = {key: 100.*correct[key]/total[key] for key in loader_keys}
        # batch_loss = train_loss/(batch_idx+1)

        progress_bar(batch_idx, min_len_loader,
                     # F"Loss: {batch_loss:.3f} | " +
                     F"Acc_CIFAR10: {acc['cifar10']:.3f} | " +
                     F"Acc_FashionMNIST: {acc['fashionmnist']:.3f} ")
    return acc, batch_loss


def test(epoch, loader):
    global best_acc, total_loss
    net.eval()
    test_loss = 0
    loader_keys = loader.keys()
    total = {key: 0 for key in loader_keys}
    correct = {key: 0 for key in loader_keys}
    len_loader = {key: len(loader[key]) for key in loader_keys}
    iter_loader = {key: iter(loader[key]) for key in loader_keys}
    min_len_loader = min(len_loader.values())

    with torch.no_grad():
        for batch_idx in range(min_len_loader):
            batch = {key: next(iter_loader[key]) for key in loader_keys}
            input = {key: batch[key][0].to(device) for key in loader_keys}
            target = {key: batch[key][1].to(device) for key in loader_keys}

            output = {}
            output['cifar10'], _ = net(input['cifar10'])
            _, output['fashionmnist'] = net(input['fashionmnist'])

            losses = {key: criterion(output[key], target[key]) for key in loader_keys}

            total_loss.eval_validloss(losses, epoch, min_len_loader)
            # loss = max(losses['cifar10'], losses['fashionmnist'])
            # loss = losses['cifar10'] + losses['fashionmnist']
            # loss = (losses['cifar10'] + losses['fashionmnist'])/2

            # test_loss += loss.item()
            predicted = {}
            for key in loader_keys:
                total[key] += target[key].size(0)
                _, predicted[key] = output[key].max(1)
                correct[key] += predicted[key].eq(target[key]).sum().item()

            acc = {key: 100. * correct[key] / total[key] for key in loader_keys}

            # batch_loss = test_loss / (batch_idx + 1)
            batch_loss = total_loss.get_valid_avgloss(epoch)

            progress_bar(batch_idx, min_len_loader,
                         # F"Loss: {batch_loss:.3f} | " +
                         F"Acc_CIFAR10: {acc['cifar10']:.3f} | " +
                         F"Acc_FashionMNIST: {acc['fashionmnist']:.3f} ")
    acc_mean = sum(acc.values())/len(acc.values())

    if acc_mean > best_acc:
        print('Saving..')
        state = {'net': net.state_dict(),
                 'acc': acc_mean,
                 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/chpt_resnet_2h.pth')
        best_acc = acc_mean
    return acc, batch_loss


for epoch in range(start_epoch, start_epoch + count_epoch):
    train(epoch, trainloader)
    test(epoch, testloader)
    print("train loss", total_loss.get_train_avgloss(epoch))
    print("validate loss", total_loss.get_valid_avgloss(epoch))
