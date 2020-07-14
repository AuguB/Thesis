import torch
import torch.nn as nn


class shuffle(nn.Module):
    def __init__(self, dim, flip=True):
        super(shuffle, self).__init__()
        self.flip = nn.Parameter(torch.tensor([flip]), requires_grad=False)
        if not self.flip:
            self.P = nn.Parameter(torch.diag(torch.ones(dim))[torch.randperm(dim)], requires_grad=False)
        else:
            self.P = nn.Parameter(torch.diag(torch.ones(dim)).flip(0), requires_grad=False)

    def forward(self, A):
        return A @ self.P, 0.

    def inverse(self, A):
        return A @ (self.P.T)


class coupling(nn.Module):
    def __init__(self, dim):
        self.max_exp = 10
        self.min_exp = -10
        super(coupling, self).__init__()
        self.dim = dim
        self.first = int(dim / 2)
        self.second = self.dim - self.first
        layers = []
        bigsize = min([1000, self.first * 3])
        sizes = [self.first] + ([bigsize] * 3) + [self.second * 2]
        for i in range(4):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], True))
            layers.append(nn.ReLU())
        layers.pop()
        self.NN = nn.Sequential(*layers)

    def forward(self, A):
        part1, part2 = A.split([self.first, self.second], 1)
        ms = self.NN(part1)
        m, s = ms.split(self.second, 1)
        s = torch.clamp(s, self.min_exp, self.max_exp)
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


class actnorm(nn.Module):
    def __init__(self, dim):
        super(actnorm, self).__init__()
        self.initiated = nn.Parameter(torch.Tensor([False]), requires_grad=False)
        self.shift = nn.Parameter(torch.full((dim,), 0.), requires_grad=True)
        self.scale = nn.Parameter(torch.full((dim,), 0.), requires_grad=True)
        self.epsilon = 0.001

    def forward(self, A):
        # This is causing issues (divide by zero stuff)
        if not self.initiated:
            with torch.no_grad():
                newshift = torch.mean(A, dim=0)
                self.shift.data = newshift
                newscale = torch.std(A - newshift, dim=0)
                newscale[newscale == 0] = 1
                newscale[newscale != newscale] = 1
                newscale[torch.isinf(newscale)] = 1
                self.scale.data = torch.pow(newscale, -1)
                self.initiated.data = torch.Tensor([True])
        self.scale.data[self.scale.data != self.scale.data] = 1
        thisScale = self.scale.abs() + self.epsilon
        A = ((A - self.shift) * thisScale)
        return A, torch.sum(torch.log(thisScale))

    def inverse(self, A):
        thisScale = self.scale.abs() + self.epsilon

        return ((A * torch.pow(thisScale, -1)) + self.shift)
