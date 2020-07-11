import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.optim import *
from plotters import show_forward, show_backward, plot_backward, mnist_noised
import matplotlib.pyplot as plt

class MNISTTrainer:
    def __init__(self):
        pass

    def train(self, net, dataset, n_epochs=300, batch_size=1, lr=1e-3, decay=1, model_signature="unknown model",
              loss_interval=10,dataname = "MNIST"):
        losses = []
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        optim = Adam(net.parameters(), lr=lr)
        scheduler = ExponentialLR(optim, decay)
        loss = 0
        total_iter = (50000/batch_size)*n_epochs
        this_iter = 0
        for e in range(n_epochs):
            for i, (v,_) in enumerate(loader):
                this_iter +=1
                print(f"\rTraining model {model_signature} {100 * (this_iter/total_iter)}% complete  ", end="")
                v = v.reshape((v.shape[0],-1))
                # v_noised = v + (2*(torch.rand(v.shape)-0.5)) * 0.2

                log_prob, _ = net(v.type(torch.FloatTensor))

                optim.zero_grad()
                loss = -log_prob

                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.6)

                optim.step()

                if torch.any(torch.isnan(loss.mean())):
                    print("found nan")

                if i % loss_interval == 0:
                    losses.append(loss.mean().detach().numpy())

                if (i % 10) == 0:
                    plot_backward(net, dataname)
                    if (i%100) == 0:
                        scheduler.step()

            if torch.any(torch.isnan(loss.mean())):
                break

            scheduler.step()

        return np.array(losses)
