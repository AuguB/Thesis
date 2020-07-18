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
        self.max_exponent = 10
        self.min_exponent = -10
        super(coupling, self).__init__()
        self.dim = dim
        self.dim_of_first_half = int(dim / 2)
        self.dim_of_second_half = self.dim - self.dim_of_first_half
        layers = []
        max_hidden_units_per_layer = min([1500, self.dim_of_first_half * 3])
        sizes = [self.dim_of_first_half] + ([max_hidden_units_per_layer] * 3) + [self.dim_of_second_half * 2]
        for i in range(4):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], True))
            layers.append(nn.ReLU())
        layers.pop()
        self.NN = nn.Sequential(*layers)

    def forward(self, A):
        first_half_in, second_half_in = A.split([self.dim_of_first_half, self.dim_of_second_half], 1)
        shift_and_scale = self.NN(first_half_in)
        shift, scale = shift_and_scale.split(self.dim_of_second_half, 1)
        scale = torch.clamp(scale, self.min_exponent, self.max_exponent)
        first_half_out = first_half_in
        second_half_out = (second_half_in * torch.exp(scale)) + shift
        output = torch.cat([first_half_out, second_half_out], 1)
        return output, torch.sum(scale, 1)

    def inverse(self, A):
        first_half_in, second_half_in = A.split([self.dim_of_first_half, self.dim_of_second_half], 1)
        shift_and_scale = self.NN(first_half_in)
        shift, scale = shift_and_scale.split(self.dim_of_second_half, 1)
        first_half_out = first_half_in
        second_half_out = (second_half_in - shift) * torch.exp(-scale)
        output = torch.cat([first_half_out, second_half_out], 1)
        return output



class actnorm(nn.Module):
    def __init__(self, dim):
        super(actnorm, self).__init__()
        self.initiated = nn.Parameter(torch.Tensor([False]), requires_grad=False)
        self.shift = nn.Parameter(torch.full((dim,), 0.), requires_grad=True)
        self.scale = nn.Parameter(torch.full((dim,), 0.), requires_grad=True)
        self.epsilon = 1e-3

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
        current_scale = self.scale.abs() + self.epsilon
        A = ((A - self.shift) * current_scale)
        return A, torch.sum(torch.log(current_scale))

    def inverse(self, A):
        current_scale = self.scale.abs() + self.epsilon
        return ((A * torch.pow(current_scale, -1)) + self.shift)
