# Multi-Task Learning in Deep Neural Networks

Этот репозиторий содержит код Multi-Task нейронной сети с Hard parameter sharing (https://ruder.io/multi-task/). 
Нейросеть использует в качестве shared net ядро нейросети ResNet:
Последний слой (full-connected) удаляется из ориганального resnet, вместо него добавляется один слой fully-connected, 
а затем еще для каждой отдельной задачи по одному fully-connected слою с количеством выходов равном количеству классов.
Каждая подзадача должна классифицировать изображение. Тренировалось на двух датасетах Cifar10 и FashionMnist.

Для нахождения баланса между двумя подзадачами (для двух лоссов) использовалось два метода: SimpleLoss(как некое среднее) и 
метод, предложенный в статье [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/)

## Запуск TRAIN
python3 train.py **flags
flags:
- --ckpt             путь к предобученной модели *.pth (по умолчанию '' -  не загружать)
- --out              директория сохранения весов (по умолчанию './checkpoint')
- --freeze_core      заморизить shared net (по умолчанию False)
- --heads            обучить конктреные "головы": both, h1, h2 (h1 - CIFAR10, h2 - FashionMNIST, both - multitask) (по умолчанию multitask)
- --dwa              использовать Dynamic Weight Average(DWA) 

### Пример: 
для трeнировки только на cifar10
python train.py --heads h1 --out chkpt_cifar10 --ckpt ./chkpt_cifar10/chpt_resnet34_cifar10.pth
для тренировки на cifar10 и FashionMNIST
python train.py --out chkpt_multitask --dwa

## Запуск TEST
python3 test.py **flags
- --ckpt             путь к предобученной модели *.pth (по умолчанию '' -  не загружать)
- --heads            вывести значение конктреной "головы": both, h1, h2 (h1 - CIFAR10, h2 - FashionMNIST, both - multitask) (по умолчанию multitask)

### Пример: 
python test.py --heads h1 --ckpt ./chkpt_cifar10/chpt_resnet34_cifar10.pth


## Эксперементы:
Share core использовалось на основе ResNet34
- Обучение только на CIFAR10: accuracity (positive_prediction/total) на валидации: 88.02
- Обучение только на FashionMNIST: accuracity (positive_prediction/total) на валидации: 93.25

Обучение multi_task  c использованием CIFAR10 и FashionMNIST: 
используя SimpleLOSS:
accuracity (positive_prediction/total) на валидации: 
- h1(CIFAR10):81.22, h2(FashionMNIST):83.33
используя DWALOSS:
accuracity (positive_prediction/total) на валидации: 
- h1(CIFAR10):80.22, h2(FashionMNIST):83.62
