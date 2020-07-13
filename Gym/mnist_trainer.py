import numpy as np
import torch
from torch.utils.data import DataLoader
from plotters import plot_backward
import math
class MNISTTrainer:
    def __init__(self):
        pass

    def train(self, net, dataset, optim, n_epochs, batch_size=1, dataname = "MNIST"):

        losses = []
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        loss = 0
        iter_per_epoch = dataset.data.shape[0]/batch_size
        loss_interval = math.floor(iter_per_epoch/100)
        total_iter = iter_per_epoch*n_epochs
        this_iter = 0
        for e in range(n_epochs):
            for i, (v,_) in enumerate(loader):
                this_iter +=1
                print(f"\rTraining model on {dataname} {100 * (this_iter/total_iter)}% complete  ", end="")
                v = v.reshape((v.shape[0],-1))
                log_prob, _ = net(v.type(torch.FloatTensor))

                optim.zero_grad()
                loss = -log_prob

                loss.mean().backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.9)

                optim.step()

                if torch.any(torch.isnan(loss.mean())):
                    print("found nan")

                if i % loss_interval == 0:
                    losses.append(loss.mean().detach().numpy())

                if (i % 10) == 0:
                    plot_backward(net, dataname)

            if torch.any(torch.isnan(loss.mean())):
                break


        return np.array(losses)
