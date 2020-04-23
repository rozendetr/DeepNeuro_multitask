import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import  *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('select device: ', device)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

means_cifar10 = [0.4914, 0.4822, 0.4465]
deviations_cifar10 = [0.2023, 0.1994, 0.2010]
means_fashionmnist = [0.5]
deviations_fashionmnist = [0.5]

transform_train_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means_cifar10, deviations_cifar10),
    ])

transform_test_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means_cifar10, deviations_cifar10),
    ])

transform_train_fashionmnist = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means_fashionmnist, deviations_fashionmnist),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # fashionMNIST have grayscale image, convert grayscale to "color"
                                                    # just repeat 1 layer to 3 layer
    ])

transform_test_fashionmnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means_fashionmnist, deviations_fashionmnist),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

batch_size_train = 64
batch_size_test = 50

trainset_cifar10 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train_cifar10)
trainloader_cifar10 = torch.utils.data.DataLoader(
    trainset_cifar10, batch_size=batch_size_train, shuffle=True, num_workers=0)
testset_cifar10 = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test_cifar10)
testloader_cifar10 = torch.utils.data.DataLoader(
    testset_cifar10, batch_size=batch_size_test, shuffle=False, num_workers=0)

trainset_fashionmnist = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform_train_fashionmnist)
trainloader_fashionmnist = torch.utils.data.DataLoader(
    trainset_fashionmnist, batch_size=batch_size_train, shuffle=True, num_workers=0)
testset_fashionmnist = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform_test_fashionmnist)
testloader_fashionmnist = torch.utils.data.DataLoader(
    testset_fashionmnist, batch_size=batch_size_test, shuffle=False, num_workers=0)

classes_cifar10 = ('plane', 'car', 'bird',
                   'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck')

classes_fashionmnist = ("T-shirt/top", "Trouser", "Pullover",
                        "Dress", "Coat", "Sandal", "Shirt",
                        "Sneaker", "Bag", "Ankle boot")

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


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    # net.freeze_shared(False)
    train_loss = 0
    total_cifar10 = 0
    total_fashionmnist = 0
    correct_cifar10 = 0
    correct_fashionmnist = 0

    len_train_cifar10 = len(trainloader_cifar10)
    len_train_fashionmnist = len(trainloader_fashionmnist)

    iter_cifar10 = iter(trainloader_cifar10)
    iter_fashionmnist = iter(trainloader_fashionmnist)

    for batch_idx in range(min(len_train_cifar10, len_train_fashionmnist)):
        batch_cifar10 = next(iter_cifar10)
        batch_fashionmnist = next(iter_fashionmnist)

        input_cifar10, target_cifar10 = batch_cifar10[0].to(device), \
                                        batch_cifar10[1].to(device)
        input_fashionmnist, target_fashionmnist = batch_fashionmnist[0].to(device), \
                                                  batch_fashionmnist[1].to(device)

        optimizer.zero_grad()

        out_cifar10, _ = net(input_cifar10)
        _, out_fashionmnist = net(input_fashionmnist)

        loss_cifar10 = criterion(out_cifar10, target_cifar10)
        loss_fashionmnist = criterion(out_fashionmnist, target_fashionmnist)

        loss = loss_cifar10 + loss_fashionmnist
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted_cifar10 = out_cifar10.max(1)
        _, predicted_fashionmnist = out_fashionmnist.max(1)
        total_cifar10 += target_cifar10.size(0)
        total_fashionmnist += target_fashionmnist.size(0)
        correct_cifar10 += predicted_cifar10.eq(target_cifar10).sum().item()
        correct_fashionmnist += predicted_fashionmnist.eq(target_fashionmnist).sum().item()

        progress_bar(batch_idx, min(len_train_cifar10, len_train_fashionmnist),
                     F"Loss: {train_loss/(batch_idx+1):.3f} | "
                     F"Acc_CIFAR10: {100.*correct_cifar10/total_cifar10:.3f} | "
                     F"Acc_FashionMNIST: {100.*correct_fashionmnist/total_fashionmnist:.3f} ")


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    total_cifar10 = 0
    total_fashionmnist = 0
    correct_cifar10 = 0
    correct_fashionmnist = 0
    len_test_cifar10 = len(testloader_cifar10)
    len_test_fashionmnist = len(testloader_fashionmnist)

    iter_cifar10 = iter(testloader_cifar10)
    iter_fashionmnist = iter(testloader_fashionmnist)

    with torch.no_grad():
        for batch_idx in range(min(len_test_cifar10, len_test_fashionmnist)):
            batch_cifar10 = next(iter_cifar10)
            batch_fashionmnist = next(iter_fashionmnist)

            input_cifar10, target_cifar10 = batch_cifar10[0].to(device), \
                                            batch_cifar10[1].to(device)
            input_fashionmnist, target_fashionmnist = batch_fashionmnist[0].to(device), \
                                                      batch_fashionmnist[1].to(device)
            out_cifar10, _ = net(input_cifar10)
            _, out_fashionmnist = net(input_fashionmnist)

            loss_cifar10 = criterion(out_cifar10, target_cifar10)
            loss_fashionmnist = criterion(out_fashionmnist, target_fashionmnist)

            loss = loss_cifar10 + loss_fashionmnist

            test_loss += loss.item()
            _, predicted_cifar10 = out_cifar10.max(1)
            _, predicted_fashionmnist = out_fashionmnist.max(1)
            total_cifar10 += target_cifar10.size(0)
            total_fashionmnist += target_fashionmnist.size(0)
            correct_cifar10 += predicted_cifar10.eq(target_cifar10).sum().item()
            correct_fashionmnist += predicted_fashionmnist.eq(target_fashionmnist).sum().item()

            progress_bar(batch_idx, min(len_test_cifar10, len_test_fashionmnist),
                         F"Loss: {test_loss / (batch_idx + 1):.3f} | "
                         F"Acc_CIFAR10: {100. * correct_cifar10 / total_cifar10:.3f} | "
                         F"Acc_FashionMNIST: {100. * correct_fashionmnist / total_fashionmnist:.3f} ")

    acc = 100. * correct_cifar10 / total_cifar10
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/chpt_resnet_2h.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)