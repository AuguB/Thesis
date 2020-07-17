import itertools
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from Flows.marginalizingflow import marginalizingFlow
from builder import get_gaussian_samples, build_px_samples
from plotters import plot_and_store_backward_pass
from utils import moving_average


def print_summary(iter, total_iter, conf, total_confs):
    print(f"\rNow evaluating model {conf}/{total_confs}, {iter / total_iter}%")


class Taster:
    def __init__(self, device, folder, taste_gaussian_models=False):
        self.device = device
        self.folder = folder
        self.loss_dict = pickle.load(open(folder + "/loss_dict.p", "rb"))
        self.model_param_dict = pickle.load(open(folder + "/param_dict.p", "rb"))
        self.train_param_dict = pickle.load(open(folder + "/info_dict.p", "rb"))
        self.taste_gaussian_models = taste_gaussian_models
        self.logli = None
        self.logli_average_over_samples = None
        self.logli_average_over_repeats = None
        if self.taste_gaussian_models:
            self.model_state_dict_dict = pickle.load(open(folder + "/model_dict.p", "rb"))

    def plot_losses(self):
        for param_tuple in self.loss_dict.keys():
            if self.taste_gaussian_models:
                name_of_model = f"GAUSS_{param_tuple[0]}dim_{param_tuple[1]}pow_{param_tuple[2]}eps_{param_tuple[3]}rep"
            else:
                name_of_data = self.model_param_dict["dataname"]
                name_of_model = f"{name_of_data}_{param_tuple[0]}eps_{param_tuple[1]}rep"
            losses_of_model = self.loss_dict[param_tuple]
            losses_lookback = 5
            plt.plot(moving_average(losses_of_model, losses_lookback))
            plt.title(name_of_model)
            plt.show()

    def compute_logli(self, precomputed=False):
        if not precomputed:
            parameter_configurations_with_indices = itertools.product(
                *[zip([j for j in range(len(i))], i) for i in self.model_param_dict.values()])
            total_configs = np.product([len(i) for i in self.model_param_dict.values()])
            for parameters_i, parameters in enumerate(parameter_configurations_with_indices):
                if self.taste_gaussian_models:
                    n_samples_from_epsilon = 200
                    n_data_samples = 2048
                    logli_buffer = np.zeros([len(i) for i in self.model_param_dict.values()] + [n_data_samples])
                    ((i, dim), (j, pow), (k, eps), (batch_i, rep)) = parameters
                    model = self.get_model(parameters)
                    data = get_gaussian_samples(n_data_samples, dim,
                                                pow).data   # Because the data is low-dim, we can do a forward pass on
                                                            # the whole set at once, and we don't need dataloader.
                                                            # That's why here the data is taken directly from the dataset.
                    logli, _ = model(data.data, marginalize=True, n_samples=n_samples_from_epsilon)
                    logli_buffer[i, j, k, batch_i, :logli.shape[0]] = logli.detach().numpy()
                else:
                    # Unpack tuple with params and indices
                    ((i, eps), (j, reps)) = parameters

                    # Get samples and construct loader
                    n_samples_from_epsilon = self.get_importance_samples(self.train_param_dict["dataname"])
                    data = build_px_samples(self.train_param_dict["dataname"])
                    n_data_samples = data.data.shape[0]
                    batch_size = 1024
                    loader = DataLoader(data, batch_size=batch_size)
                    total_iter = math.ceil(len(data) / batch_size)

                    # Construct array to hold perfs
                    logli_buffer = np.zeros([len(i) for i in self.model_param_dict.values()] + [n_data_samples])

                    # Get model
                    model = self.get_model(parameters)

                    # Compute performance by batch
                    for batch_i, batch in enumerate(loader):
                        batch = batch.to(self.device)
                        print_summary(batch_i, total_iter, parameters_i, len(total_configs))
                        current_logli, _ = model(batch, marginalize=True, n_samples=n_samples_from_epsilon)
                        logli_buffer[i, j,
                        batch_i * batch_size:(batch_i + 1) * batch_size] = current_logli.detach().cpu().numpy().copy()

            pickle.dump(logli_buffer, open(self.folder + "/logli_buffer.p", "wb"))

        self.logli = pickle.load(open(self.folder + "/logli_buffer.p", "rb"))
        self.logli_average_over_samples = np.log(np.exp(self.logli).mean(axis=-1))
        self.logli_average_over_repeats = np.log(np.exp(self.logli_average_over_samples).mean(axis=-1))

    def get_model(self, parameters):
        if self.taste_gaussian_models:
            ((i, dim), (j, pow), (k, eps), (l, rep)) = parameters
            name_of_model = f"GAUSS_{dim}dim_{pow}pow_{eps}eps_{rep}rep"
            checkpoint = torch.load(self.folder + "/" + name_of_model + ".p")
            model = marginalizingFlow(dim, eps, self.train_param_dict["n_layers"])
        else:
            ((i, eps), (j, reps)) = parameters
            name = self.train_param_dict["dataname"]
            name_of_model = f"{name}_{eps}eps_{reps}rep"
            checkpoint = torch.load(self.folder + "/" + name_of_model + ".p")
            model = marginalizingFlow(self.get_dim(name), eps, self.train_param_dict["n_layers"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model = model.to(self.device)
        return model

    def plot_avg_logli(self):
        if not self.taste_gaussian_models:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(self.model_param_dict["n_epsilons"], self.logli_average_over_repeats)
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            if len(self.model_param_dict["n_repeats"]) > 1:  # plot confidence intervals
                ci = 1.96 * np.abs(np.std(self.logli_average_over_samples, axis=1) / np.sqrt(5))
                ax.fill_between(self.model_param_dict["n_epsilons"], self.logli_average_over_repeats + ci,
                                self.logli_average_over_repeats - ci, alpha=0.1)
            plt.savefig(self.folder + "/average_logli.png")
            plt.show()
        else:
            line_styles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for j, p in enumerate(self.model_param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={p}")
                for i, d in enumerate(self.model_param_dict["n_gaussian_dims"]):
                    current_range_average_over_repeats = self.logli_average_over_repeats[i, j]
                    current_range_average_over_samples = self.logli_average_over_samples[i, j]
                    ci = 1.96 * np.abs(np.std(current_range_average_over_samples, axis=-1) / np.sqrt(5))
                    ax.plot(self.model_param_dict["n_epsilons"], current_range_average_over_repeats, label=f"{d}-D",
                            c=colors[i], alpha=1,
                            lineStyle=line_styles[i])
                    if len(self.model_param_dict["n_repeats"]) > 1:
                        ax.fill_between(self.model_param_dict["n_epsilons"], current_range_average_over_repeats + ci,
                                        current_range_average_over_repeats - ci,
                                        color=colors[i], alpha=0.1)
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if j == 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(self.folder + "/average_logli.png")
                plt.show()

    def plot_min_logli(self):
        if not self.taste_gaussian_models:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(self.model_param_dict["n_epsilons"], np.min(self.logli_average_over_samples, axis=-1))
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            plt.savefig(self.folder + "/min_logli.png")
            plt.show()
        else:
            line_styles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for power_i, power in enumerate(self.model_param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={power}")
                for dim_i, dim in enumerate(self.model_param_dict["n_gaussian_dims"]):
                    current_range_average_over_samples = self.logli_average_over_samples[dim_i, power_i]
                    ax.plot(self.model_param_dict["n_epsilons"], np.min(current_range_average_over_samples, axis=-1),
                            label=f"{dim}-D",
                            c=colors[dim_i], alpha=1,
                            lineStyle=line_styles[dim_i])
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if power_i == 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(self.folder + "/min_logli.png")
                plt.show()

    def generate(self):
        parameter_configurations_with_indices = itertools.product(
            *[zip([j for j in range(len(i))], i) for i in self.model_param_dict.values()])
        for c in parameter_configurations_with_indices:
            model = self.get_model(c)
            plot_and_store_backward_pass(self.device, model, self.train_param_dict["dataname"])

    def get_importance_samples(self, a):
        if a.endswith("MNIST") or a.endswith("CIFAR10"):
            return 10
        else:
            return 200

    def get_dim(self, a):
        if a.endswith("MNIST"):
            return 28 * 28
        elif a.endswith("CIFAR10"):
            return 3 * 32 * 32
        elif a == "HALFMOONS":
            return 2
        else:
            return 2
