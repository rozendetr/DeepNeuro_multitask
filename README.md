# Multi-Task Learning in Deep Neural Networks

Этот репозиторий содержит код Multi-Task нейронной сети с Hard parameter sharing (https://ruder.io/multi-task/). 
Нейросеть использует в качестве shared net ядро нейросети ResNet:
Последний слой (full-connected) удаляется из ориганального resnet, вместо него добавляется один слой fully-connected, 
а затем еще для каждой отдельной задачи по одному fully-connected слою с количеством выходов равном количеству классов.
Каждая подзадача должна классифицировать изображение. Тренировалось на двух датасетах Cifar10 и FashionMnist.

Для нахождения баланса между двумя подзадачами (для двух лоссов) использовалось два метода: SimpleLoss(как некое среднее) и 
метод, предложенный в статье [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704), [Shikun Liu](http://shikun.io/), [Edward Johns](https://www.robot-learning.uk/), and [Andrew Davison](https://www.doc.ic.ac.uk/~ajd/)
