import torch
import torchvision
import torchvision.transforms as transforms

# Data
print('==> Preparing data..')

means = {'cifar10': [0.4914, 0.4822, 0.4465],
         'fashionmnist': [0.5, 0.5, 0.5]}
deviations = {'cifar10': [0.2023, 0.1994, 0.2010],
              'fashionmnist': [0.5, 0.5, 0.5]}

transform_train = {}
transform_test = {}

transform_train['cifar10'] = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means['cifar10'], deviations['cifar10']),
    ])

transform_test['cifar10'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means['cifar10'], deviations['cifar10']),
    ])

transform_train['fashionmnist'] = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # fashionMNIST have grayscale image, convert grayscale to "color"
                                                    # just repeat 1 layer to 3 layer
    transforms.Normalize(means['fashionmnist'], deviations['fashionmnist']),

    ])

transform_test['fashionmnist'] = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(means['fashionmnist'], deviations['fashionmnist']),
    ])

num_workers = 0
batch_size_train = 128
batch_size_test = 50


trainset = {'cifar10': torchvision.datasets.CIFAR10(root='./data',
                                                    train=True,
                                                    download=True,
                                                    transform=transform_train['cifar10']),
            'fashionmnist': torchvision.datasets.FashionMNIST(root='./data',
                                                              train=True,
                                                              download=True,
                                                              transform=transform_train['fashionmnist'])}

trainloader = {'cifar10': torch.utils.data.DataLoader(trainset['cifar10'],
                                                      batch_size=batch_size_train,
                                                      shuffle=True,
                                                      num_workers=num_workers),
               'fashionmnist': torch.utils.data.DataLoader(trainset['fashionmnist'],
                                                           batch_size=batch_size_train,
                                                           shuffle=True,
                                                           num_workers=num_workers)}

testset = {'cifar10': torchvision.datasets.CIFAR10(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=transform_test['cifar10']),
            'fashionmnist': torchvision.datasets.FashionMNIST(root='./data',
                                                              train=False,
                                                              download=True,
                                                              transform=transform_test['fashionmnist'])}
testloader = {'cifar10': torch.utils.data.DataLoader(testset['cifar10'],
                                                      batch_size=batch_size_test,
                                                      shuffle=True,
                                                      num_workers=num_workers),
               'fashionmnist': torch.utils.data.DataLoader(testset['fashionmnist'],
                                                           batch_size=batch_size_test,
                                                           shuffle=True,
                                                           num_workers=num_workers)}

classes = {}
classes['cifar10'] = ['plane', 'car', 'bird',
                      'cat', 'deer', 'dog',
                      'frog', 'horse', 'ship', 'truck']
classes['fashionmnist'] = ["T-shirt/top", "Trouser", "Pullover",
                           "Dress", "Coat", "Sandal", "Shirt",
                           "Sneaker", "Bag", "Ankle boot"]