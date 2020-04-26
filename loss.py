import torch
import torch.nn as nn
import numpy as np


class DWALoss(object):

    def __init__(self, count_loss, count_epoch, T=2.0):
        self.count_loss = count_loss
        self.count_epoch = count_epoch
        self.T = T
        self.avg_cost = np.zeros([count_epoch, 2*count_loss], dtype=np.float32)
        # for every epoch avg_cost save two part of costs: costs of train and costs of valid

        self.lambda_weight = np.ones([count_loss, count_epoch])
        self.weight = np.zeros(count_loss, dtype=np.float32)

    def eval_lambda_weight(self, index_epoch):
        """
        Evalute coeff of weighs for every losses
        :param index_epoch:
        :return:
        """

        if index_epoch == 0 or index_epoch == 1:
            self.lambda_weight[:, index_epoch] = 1.0
        else:
            self.weight = self.avg_cost[index_epoch-1, :self.count_loss] / self.avg_cost[index_epoch-2, self.count_loss:]
            sum_exp = np.sum(np.exp(self.weight/self.T))
            self.lambda_weight[:, index_epoch] = 3*np.exp(self.weight / self.T) / sum_exp

    def eval_trainloss(self, train_losses, index_epoch, count_train_batch):
        """
        Evalute total loss for multitask loss and refresh train_part of avg_cost
        :param train_losses: is dictionary of losses
        :param index_epoch:
        :param count_train_batch: counts of train batches
        :return:
        """
        losses = [train_losses[key] for key in train_losses.keys()]
        # losses = train_losses.values()


        total_loss = torch.mean(sum(self.lambda_weight[i, index_epoch] * losses[i] for i in range(len(losses))))
        costs = np.array([loss.item() for loss in losses],  dtype=np.float32)
        self.avg_cost[index_epoch, :self.count_loss] += costs / count_train_batch
        return total_loss

    def eval_validloss(self, valid_losses, index_epoch, count_valid_batch):
        """
        Refresh valid_part 0f avg_cost
        :param valid_losses: is dictionary of losses
        :param index_epoch:
        :param count_valid_batch:counts of validate batches
        :return:
        """
        # losses = valid_losses.values()
        losses = [valid_losses[key] for key in valid_losses.keys()]
        costs = np.array([loss.item() for loss in losses],  dtype=np.float32)
        self.avg_cost[index_epoch, self.count_loss:] += costs / count_valid_batch

    def get_train_avgloss(self, index_epoch):
        return self.avg_cost[index_epoch, :self.count_loss]

    def get_valid_avgloss(self, index_epoch):
        return self.avg_cost[index_epoch, self.count_loss:]


class SimpleLoss(object):

    def __init__(self, count_loss, count_epoch, device, T=-0.5):
        self.count_loss = count_loss
        self.count_epoch = count_epoch
        self.T = T

        logsigma = [T for loss in range(count_loss)]
        self.logsigma = nn.Parameter(torch.FloatTensor(logsigma).to(device))

        self.avg_cost = np.zeros([count_epoch, 2*count_loss], dtype=np.float32)
        # for every epoch avg_cost save two part of costs: costs of train and costs of valid

    def eval_trainloss(self, train_losses, index_epoch, count_train_batch):
        """
        Evalute total loss for multitask loss and refresh train_part of avg_cost
        :param train_losses: is dictionary of losses
        :param index_epoch:
        :param count_train_batch: counts of train batches
        :return:
        """
        losses = [train_losses[key] for key in train_losses.keys()]
        total_loss = sum(1 / (2 * torch.exp(self.logsigma[i])) * losses[i] + self.logsigma[i] / 2 for i in range(len(losses)))

        costs = np.array([loss.item() for loss in losses],  dtype=np.float32)
        self.avg_cost[index_epoch, :self.count_loss] += costs / count_train_batch
        return total_loss

    def eval_validloss(self, valid_losses, index_epoch, count_valid_batch):
        """
        Refresh valid_part 0f avg_cost
        :param valid_losses: is dictionary of losses
        :param index_epoch:
        :param count_valid_batch:counts of validate batches
        :return:
        """
        losses = [valid_losses[key] for key in valid_losses.keys()]
        costs = np.array([loss.item() for loss in losses],  dtype=np.float32)
        self.avg_cost[index_epoch, self.count_loss:] += costs / count_valid_batch

    def get_train_avgloss(self, index_epoch):
        return self.avg_cost[index_epoch, :self.count_loss]

    def get_valid_avgloss(self, index_epoch):
        return self.avg_cost[index_epoch, self.count_loss:]


