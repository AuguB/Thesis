import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.optim import *

from plotters import show_forward, show_backward


class Trainer:
    def __init__(self):
        pass

    def train(self, net, dataset, n_epochs=300, batch_size=1, lr=1e-3, decay=1, model_signature="unknown model", loss_interval = 10):
        # Store the losses
        losses = []
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        optim = Adam(net.parameters(), lr=lr)

        scheduler = ExponentialLR(optim, decay)
        loss = 0
        for e in range(n_epochs):
            for i, v in enumerate(loader):

                log_prob, _ = net(v.type(torch.FloatTensor), marginalize = False)
                optim.zero_grad()
                loss = -log_prob
                loss.mean().backward()
                optim.step()

                if torch.any(torch.isnan(loss.mean())):
                    print("found nan")
                    break

                if i%loss_interval == 0:
                    losses.append(loss.mean().detach().numpy())

            if torch.any(torch.isnan(loss.mean())):
                break
            # if e%8 == 0:
            #     show_forward(dataset, net, "")
            #     show_backward(net, "")
            print(f"\rTraining model {model_signature} {round(100*((e + 1) / n_epochs))}% complete  ", end="")
            scheduler.step()

        return np.array(losses)