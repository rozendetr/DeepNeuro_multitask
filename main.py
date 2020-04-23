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
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader_cifar10):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out_1, out_2 = net(inputs)
        loss_1 = criterion(out_1, targets)
        loss_1.backward()
        optimizer.step()

        train_loss += loss_1.item()
        _, predicted = out_1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader_cifar10), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)