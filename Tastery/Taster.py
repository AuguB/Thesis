import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

from Flows.marginalizingflow import marginalizingFlow
from builder import get_gaussian_samples, build_px_samples
from utils import moving_average


class Taster:
    def __init__(self, folder, gaussian = False):
        self.folder = folder
        self.loss_dict = pickle.load(open(folder+"/loss_dict.p","rb"))
        self.param_dict = pickle.load(open(folder+"/param_dict.p","rb"))
        self.info_dict = pickle.load(open(folder+"/info_dict.p","rb"))
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


    def compute_neglogli(self, precomputed = False):
        if not precomputed:
            n_datapoints = 2048
            perfs = np.zeros([len(i) for i in self.param_dict.values()]+[n_datapoints])
            print(self.param_dict.values())
            configs = itertools.product(*[zip([j for j in range(len(i))],i) for i in self.param_dict.values()])
            for c in configs:
                print(c)
                if self.gaussian:
                    n_importance_samples = 200
                    ((i,dim), (j,pow), (k,eps), (l,rep)) = c
                    title = f"GAUSS_{dim}dim_{pow}pow_{eps}eps_{rep}rep"
                    checkpoint = torch.load(self.folder+"/"+title+".p")
                    data = get_gaussian_samples(n_datapoints, dim, pow)
                    model = marginalizingFlow(dim, eps, self.info_dict["n_layers"])
                    model.load_state_dict(checkpoint["model"])
                    logli, _ = model(data.data, marginalize=True,n_samples = n_importance_samples)
                    perfs[i,j,k,l,:logli.shape[0]] = logli.detach().numpy()
                else:
                    n_importance_samples = self.get_importance_samples(self.info_dict["dataname"])
                    ((i,eps), (j,reps)) = c
                    name = self.param_dict["dataname"]
                    title = f"{name}_{eps}eps_{reps}rep"
                    checkpoint = torch.load(self.folder+"/"+title+".p")
                    data = build_px_samples(self.info_dict["dataname"],n_samples = n_datapoints)
                    dim = data.data.shape[0]
                    model = marginalizingFlow(dim, eps, self.info_dict["n_layers"])
                    model.load_state_dict(checkpoint["model"])
                    logli, _ = model(data, marginalize=True, n_samples=n_importance_samples)
                    perfs[i, j, :logli.shape[0]] = logli.detach().numpy()
            pickle.dump(perfs, open(self.folder+"/perfs.p", "wb"))

        self.perfs = pickle.load(open(self.folder+"/perfs.p", "rb"))
        self.avg_logli_of_samples = np.log(np.exp(self.perfs).mean(axis=-1))
        self.avg_logli_of_repeats = np.log(np.exp(self.avg_logli_of_samples).mean(axis=-1))

    def plot_avg_logli(self):
        if not self.gaussian:
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            ax.plot(self.param_dict["n_epsilons"], self.avg_logli_of_repeats)
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            if len(self.param_dict["n_repeats"]) > 1: # plot confidence intervals
                ci = 1.96 * np.abs(np.std(self.avg_logli_of_samples, axis=1) / np.sqrt(5))
                ax.fill_between(self.param_dict["n_epsilons"], self.avg_logli_of_repeats + ci, self.avg_logli_of_repeats - ci, alpha = 0.1)
            plt.savefig(self.folder+"/average_logli.png")
            plt.show()
        else:
            lineStyles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for j, p in enumerate(self.param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={p}")
                for i, d in enumerate(self.param_dict["n_gaussian_dims"]):
                    currentRange = self.avg_logli_of_repeats[i,j]
                    currentRange_avgofsamples = self.avg_logli_of_samples[i,j]
                    ci = 1.96 * np.abs(np.std(currentRange_avgofsamples, axis=-1) / np.sqrt(5))
                    ax.plot(self.param_dict["n_epsilons"], currentRange, label=f"{d}-D", c=colors[i], alpha=1, lineStyle=lineStyles[i])
                    if len(self.param_dict["n_repeats"]) > 1:
                        ax.fill_between(self.param_dict["n_epsilons"], currentRange+ci,currentRange-ci, color =colors[i], alpha = 0.1)
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
            ax.plot(self.param_dict["n_epsilons"], np.min(self.avg_logli_of_samples,axis=-1))
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
                    ax.plot(self.param_dict["n_epsilons"], np.min(currentRange_avgofsamples, axis= -1), label=f"{d}-D", c=colors[i], alpha=1,
                            lineStyle=lineStyles[i])
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if j == 0:
                    plt.legend()
                plt.tight_layout()
                plt.savefig(self.folder + "/min_logli.png")
                plt.show()

    def generate(self):
        pass

    def get_importance_samples(self, a):
        if a.endswith("MNIST") or a.endswith("CIFAR10"):
            return 10
        else:
            return 200