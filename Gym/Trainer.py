import numpy as np
import torch
from torch.utils.data import DataLoader
import math

from plotters import plot_and_store_backward_pass


def train(device, flow, dataset, optim, n_epochs, name_of_data="MNIST", batch_size=1,
          clip_norm: float = None, make_plots=False, print_status = False):

    flow = flow.to(device)
    list_of_losses = []
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    number_of_iterations_per_epoch = dataset.data.shape[0] / batch_size
    number_of_loss_samples_per_epoch = 100
    loss_sample_interval = max(math.floor(number_of_iterations_per_epoch / number_of_loss_samples_per_epoch), 1)
    total_number_of_iterations = number_of_iterations_per_epoch * n_epochs
    current_iteration = 0

    for current_epoch in range(n_epochs):
        for current_batch_i, current_batch in enumerate(dataloader):
            current_iteration += 1
            current_batch = current_batch.to(device)
            if print_status:
                print(
                    f"\rTraining model on {name_of_data} {round(100 * (current_iteration / total_number_of_iterations), 2)}% complete   ",
                    end="")
            log_prob_of_current_batch, _ = flow(current_batch)
            optim.zero_grad()
            loss = -log_prob_of_current_batch
            loss.mean().backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 0.4)
            optim.step()
            if current_batch_i % loss_sample_interval == 0:
                list_of_losses.append(loss.mean().cpu().detach().numpy())

        if make_plots:
            plot_and_store_backward_pass(flow, name_of_data)
        if torch.any(torch.isnan(loss.mean())):
            print("found nan")
            break

    return np.array(list_of_losses)


class Trainer:
    def __init__(self):
        pass
