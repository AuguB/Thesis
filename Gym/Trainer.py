import numpy as np
import torch
from torch.utils.data import DataLoader
import math

from plotters import plot_backward


class Trainer:
    def __init__(self):
        pass

    def train(self, device, net, dataset, optim, n_epochs, dataname="MNIST", batch_size=1,
              clipNorm: float = None, make_plots=False):

        net.to(device)
        losses = []
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        iter_per_epoch = dataset.data.shape[0] / batch_size
        loss_interval = max(math.floor(iter_per_epoch / 100), 1)
        total_iter = iter_per_epoch * n_epochs
        this_iter = 0

        for e in range(n_epochs):
            print(f"epoch {e} of {n_epochs} for {dataname}")
            for i, v in enumerate(loader):
                this_iter += 1
                v.to(device)
                # print(f"Training model on {dataname} {100 * (this_iter / total_iter)}% complete  ", end="")
                log_prob, _ = net(v)
                optim.zero_grad()
                loss = -log_prob
                loss.mean().backward()
                if clipNorm:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.4)
                optim.step()
                if i % loss_interval == 0:
                    losses.append(loss.mean().detach().numpy())

            if make_plots:
                plot_backward(net, dataname)
            if torch.any(torch.isnan(loss.mean())):
                # print("found nan")
                break

        return np.array(losses)
