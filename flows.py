import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from utils import createMaskSet, show_connections


class coupling(nn.Module):
    def __init__(self, dim):
        super(coupling, self).__init__()
        self.dim = dim
        self.first = int(math.floor(dim / 2))
        self.second = self.dim - self.first
        layers = []
        sizes = [self.first] + ([self.first * 3] * 3) + [self.second * 2]
        for i in range(4):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], True))
            layers.append(nn.ReLU())
        layers.pop()
        self.NN = nn.Sequential(*layers)

    def forward(self, A):
        part1, part2 = A.split([self.first, self.second], 1)
        ms = self.NN(part1)
        m, s = ms.split(self.second, 1)
        outputpart1 = part1
        outputpart2 = (part2 * torch.exp(s)) + m
        output = torch.cat([outputpart1, outputpart2], 1)
        return output, torch.sum(s, 1)

    def inverse(self, A):
        part1, part2 = A.split([self.first, self.second], 1)
        ms = self.NN(part1)
        m, s = ms.split(self.second, 1)
        outputpart1 = part1
        outputpart2 = (part2 - m) * torch.exp(-s)
        output = torch.cat([outputpart1, outputpart2], 1)
        return output


class shuffle(nn.Module):
    def __init__(self, dim, flip=True):
        super(shuffle, self).__init__()
        self.flip = nn.Parameter(torch.tensor([flip]), requires_grad=False)
        if not  self.flip:
            self.P = nn.Parameter(torch.diag(torch.ones(dim))[torch.randperm(dim)], requires_grad=False)
        else:
            self.P = nn.Parameter(torch.diag(torch.ones(dim)).flip(0), requires_grad = False)

    def forward(self, A):
        return A @ self.P, 0.

    def inverse(self, A):
        return A @ (self.P.T)

class actnorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.initiated = nn.Parameter(torch.tensor([False]), requires_grad=False)
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones((1, dim)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)
        self.epsilon = 1e-8

    def forward(self, A):
        if not self.initiated:
            new_beta = A.mean(dim=0).detach().view((1, self.dim)).detach()
            self.beta.data = new_beta

            new_alpha = torch.std((A - self.beta), dim=0)
            new_alpha[new_alpha == 0] = 1.0
            new_alpha = (new_alpha ** (-2)).view((1, self.dim)).log().detach()  # Take the inverse
            self.alpha.data = new_alpha
            self.initiated.data = torch.tensor([True])

        A_normed = (A-self.beta) * (self.alpha.exp()+self.epsilon)
        return A_normed, (self.alpha.exp() + self.epsilon).log().sum(dim=1)

    def inverse(self, A):
        return (A / (self.alpha.exp()+self.epsilon))+self.beta


