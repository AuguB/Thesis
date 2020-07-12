import itertools
import pickle
import torch
from builder import get_gaussian_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

from utils import moving_average, reject_outliers

folder = "/home/guus/PycharmProjects/Thesis/Runs/gaussian4_2020-07-03 14:32:18.165834"
loss_dict = pickle.load(open(folder + "/loss_dict.p", "rb"))
model_state_dict_dict = pickle.load(open(folder + "/model_dict.p", "rb"))
param_dict = pickle.load(open(folder + "/param_dict.p", "rb"))
model_dict = {}

n_epsilons = param_dict["n_epsilons"]
n_gaussian_dims = param_dict["n_gaussian_dims"]
n_repeats = param_dict["n_repeats"]
gaussian_max_shifts = param_dict["gaussian_max_shifts"]
gaussian_powers = param_dict["gaussian_powers"]
reps = [i for i in range(n_repeats)]

model_tups = itertools.product(n_gaussian_dims, n_epsilons, gaussian_max_shifts, gaussian_powers, reps)
loss_tups = itertools.product(n_gaussian_dims, n_epsilons, gaussian_max_shifts, gaussian_powers)

for param in param_dict.keys():
    print(f"{param}, {param_dict[param]}")

for tup in model_tups:
    model_dict[tup] = marginalizingFlow(tup[0], tup[1], 4)
    for i in ["normalizingFlow.parts.0.initiated", "normalizingFlow.parts.5.initiated", "normalizingFlow.parts.10.initiated", "normalizingFlow.parts.15.initiated"]:
        model_state_dict_dict[tup][i] = torch.Tensor([True])
    model_dict[tup].load_state_dict(model_state_dict_dict[tup])

    model_dict[tup].eval()

with torch.no_grad():
    #
    # # Plot the losses
    # for tup in loss_tups:
    #     model_signature = f"gauss_{tup[0]}dim_{tup[1]}eps_{tup[2]}shift_{tup[3]}pow"
    #     losses = loss_dict[tup]
    #     losses_lookback = 20
    #     plt.plot(moving_average(loss_dict[tup].mean(axis = 0), losses_lookback))
    #     plt.title(model_signature)
    #     plt.show()
    #
    # Store the performances
    n_samples = 200
    # perfs = np.zeros(
    #     (len(n_gaussian_dims), len(n_epsilons), len(gaussian_max_shifts), len(gaussian_powers), len(reps), n_samples))
    # total_perfs = len(n_gaussian_dims) * len(n_epsilons) * len(gaussian_max_shifts) * len(gaussian_powers) * len(reps)
    # r = 0
    #
    # for dim_i, gaussian_dim in enumerate(n_gaussian_dims):
    #     for shift_i, shift in enumerate(gaussian_max_shifts):
    #         for pow_i, pow in enumerate(gaussian_powers):
    #             y = get_gaussian_samples(n_samples, gaussian_dim, shift, pow)
    #             for eps_i, epsilon in enumerate(n_epsilons):
    #
    #                 for rep in reps:
    #                     r += 1
    #                     print(f"\r{r / total_perfs}", end="")
    #                     ll, transformed = model_dict[(gaussian_dim, epsilon, shift, pow, rep)](y, marginalize=True, n_samples = n_samples)
    #                     # nll_bitsperdim = -((ll / 2)) / np.log(2)
    #                     # Remove outliers
    #                     perfs[dim_i, eps_i, shift_i, pow_i, rep] = ll.detach().numpy()
    # print()
    # pickle.dump(perfs, open(f"/home/guus/PycharmProjects/Thesis/Perfs/Gaussian_perfs{n_samples}.p", "wb"))
    perfs = pickle.load(open(f"/home/guus/PycharmProjects/Thesis/Perfs/Gaussian_perfs{n_samples}.p", "rb"))

    perfs_samav = np.log(np.exp(perfs).mean(axis=-1))
    perfs_repav = np.log(np.exp(perfs_samav).mean(axis=-1))

    dims, eps, shifts, pows, reps, _ = perfs.shape


    # Performance plots
    # Group = dim
    # fix: shift, pow
    lineStyles = ["-","--","-."]
    colors = ["black", "black","black"]
    print("D & a & B=0 & B=1 & B=2 & B=3 & B=4 \\\\\\hline")
    for j, p in enumerate(gaussian_powers):
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

        ax.set_title(f"ψ={p}")
        for i,d in enumerate(n_gaussian_dims):
            currentRangeSamav = -perfs_samav[i, :, 0, j]
            currentRangeRepav = -perfs_repav[i, :, 0, j]
            minRange = np.min(currentRangeSamav, axis=-1)

            print(f"{d} & {p} & "+" & ".join([str(round(s,2)) for s in minRange]), end = "\\\\")
            if i == 2:
                print("\\hline")
            else:
                print()

            # ax.plot(n_epsilons, -perfs_repav[i, :, 0, j], label = f"{d}-D", c=colors[i], alpha = 1)
            ci = 1.96 * np.abs(np.std(currentRangeSamav, axis=1)/np.sqrt(5))

            ax.plot(n_epsilons, minRange, label = f"{d}-D", c=colors[i], alpha = 1, lineStyle = lineStyles[i])
            # ax.fill_between(n_epsilons, currentRangeRepav+ci,currentRangeRepav-ci, color =colors[i], alpha = 0.1)
            # for r in range(n_repeats):
            #     ax.plot(n_epsilons, -perfs_samav[i, :, 0, j,r],c=colors[i],alpha=0.2)

        ax.set_xlabel("B")
        ax.set_ylabel("Negative Log-likelihood")
        if j == 2:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot1_{p}.png")
        plt.show()

    lineStyles = ["-", "--", "-."]
    colors = ["red", "blue", "green"]
    for j, p in enumerate(gaussian_powers):
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        ax.set_title(f"ψ={p}")
        for i, d in enumerate(n_gaussian_dims):
            currentRangeSamav = -perfs_samav[i, :, 0, j]
            currentRangeRepav = -perfs_repav[i, :, 0, j]
            minRange = np.min(currentRangeSamav, axis=-1)
            # ax.plot(n_epsilons, -perfs_repav[i, :, 0, j], label = f"{d}-D", c=colors[i], alpha = 1)
            ci = 1.96 * np.abs(np.std(currentRangeSamav, axis=1) / np.sqrt(5))
            ax.plot(n_epsilons, currentRangeRepav, label=f"{d}-D", c=colors[i], alpha=1, lineStyle=lineStyles[i])
            ax.fill_between(n_epsilons, currentRangeRepav+ci,currentRangeRepav-ci, color =colors[i], alpha = 0.1)

            # for r in range(n_repeats):
            #     ax.plot(n_epsilons, -perfs_samav[i, :, 0, j,r],c=colors[i],alpha=0.2,lineStyle=lineStyles[i])

        ax.set_xlabel("B")
        ax.set_ylabel("Negative Log-likelihood")
        if j == 2:
            plt.legend()
        plt.tight_layout()
        plt.savefig(f"/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot4_{p}.png")
        plt.show()

    # dims,eps,shifts,pows,reps,_ = perfs.shape
    # Plot 3:
    # inverse plots
    n_generations = 2000

    fig, ax = plt.subplots( len(gaussian_powers),len(n_epsilons)+1, figsize=(12, 6))
    for i, e in enumerate(n_epsilons):
        for j, p in enumerate(gaussian_powers):
            ax[j, i].set_title(f"ψ={p}, B={e}")
            if p % 2 == 1:
                ax[j, i].set_xlim((-3 ** p, 3 ** p))
                ax[j, i].set_ylim((-3 ** p, 3 ** p))
            else:
                ax[j, i].set_xlim((-0.5, 3 ** p))
                ax[j, i].set_ylim((-0.5, 3 ** p))
            data = MultivariateNormal(loc=torch.zeros(model_dict[(2, e, 0, p, 0)].Q),
                                      covariance_matrix=torch.diag(torch.ones(model_dict[(2, e, 0, p, 0)].Q))).sample(
                (n_generations,))
            inverse = model_dict[(2, e, 0, p, 0)].inverse(data).detach().numpy()
            ax[j, i].scatter(inverse[:, 0], inverse[:, 1], s=1, c="black",alpha = 0.5)
    for j,p in enumerate(gaussian_powers):
        i = len(n_epsilons)
        if p % 2 == 1:
            ax[j, i].set_xlim((-3 ** p, 3 ** p))
            ax[j, i].set_ylim((-3 ** p, 3 ** p))
        else:
            ax[j, i].set_xlim((-0.5, 3 ** p))
            ax[j, i].set_ylim((-0.5, 3 ** p))
        data = get_gaussian_samples(n_generations, 2, 0, p)
        ax[j,i].set_title(f"target for ψ={p}")
        ax[j,i].scatter(data[:,0],data[:,1],c="red",s=1,alpha=0.5)
    # ax[0].set_ylabel("-Ll (b/d)")
    # ax[0].set_xlabel("B")
    plt.legend()
    plt.tight_layout()

    plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot3.png")
    plt.show()

    # dims,eps,shifts,pows,reps,_ = perfs.shape

# Correlation
for i,p in enumerate(gaussian_powers):
    for j, d in enumerate(n_gaussian_dims):

        # obs = np.stack([perfs_repav[j,:,0,i],np.array(n_epsilons)], axis=1).T
        print(f"power:{p},dim{d}")
        print(np.corrcoef(perfs_repav[j,:,0,i],np.array(n_epsilons))[1,0])
