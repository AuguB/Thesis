from datetime import datetime

import torch
import numpy as np

class SoftplusInv(torch.nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)
        self.delta = torch.nn.Parameter(torch.Tensor([0.001]), requires_grad=False)

    def forward(self,A):
        return super().forward(A)+self.delta

    def inverse(self,A):
        return torch.log(torch.exp(self.beta*(A-self.delta)-1))/self.beta

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    result = np.where(s<m, data, np.median(data))
    return result

def createMaskSet(D, order_agnostic):

    # Sample m vectors
    L = len(D)
    if order_agnostic:
        m = [torch.randperm(D[0])+1]
    else:
        m = [torch.arange(D[0]) + 1]

    for l in range(1, L):
        if(min(m[l-1] >= D[l])):
            availablerange=[D[l]-1]
        else:
            availablerange = np.arange(min(m[l-1]), D[l])
        m.append(torch.from_numpy(np.random.choice(availablerange, D[l])))

    # Constructing masks for each layer
    M = []
    for l in range(1, L-1):
        M.append(1. * (m[l].unsqueeze(-1) >= m[l - 1].unsqueeze(0)))
    M.append((1. * (m[0].unsqueeze(-1) > m[-2].unsqueeze(0))).repeat((2,1)))
    return M


def show_connections(maskSet):
    M_c = maskSet[-1].clone()
    print(M_c)
    for i in reversed(maskSet[:-1]):
        print(i)
        M_c = M_c@i
    print(M_c)

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return img.view(self.new_size)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Flattener(object):
    def __call__(self,tensor):
        return tensor.view((1,-1)).squeeze()

    def __repr__(self):
        return self.__class__.__name__


def timecode():
    time = datetime.now()
    return f"{time.year}-{time.month}-{time.day}_{time.hour}:{time.minute}:{time.second}"