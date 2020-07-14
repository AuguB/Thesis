import itertools
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from Flows.marginalizingflow import marginalizingFlow
from builder import get_gaussian_samples, build_px_samples
from plotters import plot_backward
from utils import moving_average


def print_summary(iter, total_iter, conf, total_confs):
    print(f"\rNow evaluating model {conf}/{total_confs}, {iter / total_iter}%")


class Taster:
    def __init__(self, folder, gaussian=False):
        self.folder = folder
        self.loss_dict = pickle.load(open(folder + "/loss_dict.p", "rb"))
        self.param_dict = pickle.load(open(folder + "/param_dict.p", "rb"))
        self.info_dict = pickle.load(open(folder + "/info_dict.p", "rb"))
        self.gaussian = gaussian
        if self.gaussian:
            self.model_state_dict_dict = pickle.load(open(folder + "/model_dict.p", "rb"))

    def plot_losses(self):
        for tup in self.loss_dict.keys():
            if self.gaussian:
                title = f"GAUSS_{tup[0]}dim_{tup[1]}pow_{tup[2]}eps_{tup[3]}rep"
            else:
                name = self.param_dict["dataname"]
                title = f"{name}_{tup[0]}eps_{tup[1]}rep"
            losses = self.loss_dict[tup]
            losses_lookback = 5
            plt.plot(moving_average(losses, losses_lookback))
            plt.title(title)
            plt.show()

    def compute_neglogli(self, precomputed=False):
        if not precomputed:
            configs = itertools.product(*[zip([j for j in range(len(i))], i) for i in self.param_dict.values()])
            total_configs = np.product([len(i) for i in self.param_dict.values()])
            for ci, c in enumerate(configs):
                if self.gaussian:
                    n_importance_samples = 200
                    n_datapoints = 2048
                    perfs = np.zeros([len(i) for i in self.param_dict.values()] + [n_datapoints])
                    ((i, dim), (j, pow), (k, eps), (l, rep)) = c
                    model = self.get_model(c)
                    data = get_gaussian_samples(n_datapoints, dim, pow)
                    logli, _ = model(data.data, marginalize=True, n_samples=n_importance_samples)
                    perfs[i, j, k, l, :logli.shape[0]] = logli.detach().numpy()
                else:
                    # Unpack tuple with params and indices
                    ((i, eps), (j, reps)) = c

                    # Get samples and construct loader
                    n_importance_samples = self.get_importance_samples(self.info_dict["dataname"])
                    data = build_px_samples(self.info_dict["dataname"])
                    n_datapoints = data.data.shape[0]
                    batch_size = 1024
                    loader = DataLoader(data, batch_size=batch_size)
                    total_iter = math.ceil(len(data) / batch_size)

                    # Construct array to hold perfs
                    perfs = np.zeros([len(i) for i in self.param_dict.values()] + [n_datapoints])

                    # Get model
                    model = self.get_model(c)

                    # Compute performance by batch
                    for l, v in enumerate(loader):
                        print_summary(l, total_iter, ci, len(total_configs))
                        c_logli, _ = model(v, marginalize=True, n_samples=n_importance_samples)
                        perfs[i,j,l * batch_size:(l + 1) * batch_size] = c_logli.detach().numpy().copy()

            pickle.dump(perfs, open(self.folder + "/perfs.p", "wb"))

        self.perfs = pickle.load(open(self.folder + "/perfs.p", "rb"))
        self.avg_logli_of_samples = np.log(np.exp(self.perfs).mean(axis=-1))
        self.avg_logli_of_repeats = np.log(np.exp(self.avg_logli_of_samples).mean(axis=-1))

    def get_model(self, c):
        if self.gaussian:
            ((i, dim), (j, pow), (k, eps), (l, rep)) = c
            title = f"GAUSS_{dim}dim_{pow}pow_{eps}eps_{rep}rep"
            checkpoint = torch.load(self.folder + "/" + title + ".p")
            model = marginalizingFlow(dim, eps, self.info_dict["n_layers"])
        else:
            ((i, eps), (j, reps)) = c
            name = self.info_dict["dataname"]
            title = f"{name}_{eps}eps_{reps}rep"
            checkpoint = torch.load(self.folder + "/" + title + ".p")
            model = marginalizingFlow(self.get_dim(name), eps, self.info_dict["n_layers"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    def plot_avg_logli(self):
        if not self.gaussian:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(self.param_dict["n_epsilons"], self.avg_logli_of_repeats)
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            if len(self.param_dict["n_repeats"]) > 1:  # plot confidence intervals
                ci = 1.96 * np.abs(np.std(self.avg_logli_of_samples, axis=1) / np.sqrt(5))
                ax.fill_between(self.param_dict["n_epsilons"], self.avg_logli_of_repeats + ci,
                                self.avg_logli_of_repeats - ci, alpha=0.1)
            plt.savefig(self.folder + "/average_logli.png")
            plt.show()
        else:
            lineStyles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for j, p in enumerate(self.param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={p}")
                for i, d in enumerate(self.param_dict["n_gaussian_dims"]):
                    currentRange = self.avg_logli_of_repeats[i, j]
                    currentRange_avgofsamples = self.avg_logli_of_samples[i, j]
                    ci = 1.96 * np.abs(np.std(currentRange_avgofsamples, axis=-1) / np.sqrt(5))
                    ax.plot(self.param_dict["n_epsilons"], currentRange, label=f"{d}-D", c=colors[i], alpha=1,
                            lineStyle=lineStyles[i])
                    if len(self.param_dict["n_repeats"]) > 1:
                        ax.fill_between(self.param_dict["n_epsilons"], currentRange + ci, currentRange - ci,
                                        color=colors[i], alpha=0.1)
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if j == 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(self.folder + "/average_logli.png")
                plt.show()

    def plot_min_logli(self):
        if not self.gaussian:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot(self.param_dict["n_epsilons"], np.min(self.avg_logli_of_samples, axis=-1))
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            plt.savefig(self.folder + "/min_logli.png")
            plt.show()
        else:
            lineStyles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for j, p in enumerate(self.param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={p}")
                for i, d in enumerate(self.param_dict["n_gaussian_dims"]):
                    currentRange_avgofsamples = self.avg_logli_of_samples[i, j]
                    ax.plot(self.param_dict["n_epsilons"], np.min(currentRange_avgofsamples, axis=-1), label=f"{d}-D",
                            c=colors[i], alpha=1,
                            lineStyle=lineStyles[i])
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if j == 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(self.folder + "/min_logli.png")
                plt.show()

    def generate(self):
        configs = itertools.product(*[zip([j for j in range(len(i))], i) for i in self.param_dict.values()])
        for c in configs:
            model = self.get_model(c)
            plot_backward(model, self.info_dict["dataname"])

    def get_importance_samples(self, a):
        if a.endswith("MNIST") or a.endswith("CIFAR10"):
            return 10
        else:
            return 200

    def get_dim(self, a):
        if a.endswith("MNIST"):
            return 28*28
        elif a.endswith("CIFAR10"):
            return 3*32*32
        elif a == "HALFMOONS":
            return 2
        else:
            return 2
