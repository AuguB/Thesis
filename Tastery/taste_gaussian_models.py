import itertools
import pickle
import torch
from builder import get_gaussian_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from utils import moving_average, reject_outliers

folder = "/home/guus/PycharmProjects/Thesis/Runs/gaussian6_2020-06-26 17:53:30.250887"
loss_dict = pickle.load(open(folder+"/loss_dict.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
param_dict = pickle.load(open(folder+"/param_dict.p","rb"))
model_dict = {}

n_epsilons = param_dict["n_epsilons"]
n_gaussian_dims = param_dict["n_gaussian_dims"]
n_repeats = param_dict["n_repeats"]
gaussian_max_shifts = param_dict["gaussian_max_shifts"]
gaussian_powers = param_dict["gaussian_powers"]
reps = [ i for i in range(n_repeats)]

model_tups = itertools.product(n_gaussian_dims, n_epsilons, gaussian_max_shifts, gaussian_powers, reps)
loss_tups = itertools.product(n_gaussian_dims, n_epsilons, gaussian_max_shifts, gaussian_powers)

for param in param_dict.keys():
    print(f"{param}, {param_dict[param]}")

for tup in model_tups:
    N = tup[0]
    M = tup[1]
    model_dict[tup] = marginalizingFlow(N, M,6)
    model_dict[tup].load_state_dict(model_state_dict_dict[tup])
    model_dict[tup].eval()
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
# n_samples = 300
# perfs = np.zeros((len(n_gaussian_dims), len(n_epsilons), len(gaussian_max_shifts), len(gaussian_powers), len(reps)))
# total_perfs = len(n_gaussian_dims)* len(n_epsilons)* len(gaussian_max_shifts)* len(gaussian_powers)* len(reps)
# r = 0
# for dim_i, gaussian_dim in enumerate(n_gaussian_dims):
#     for shift_i, shift in enumerate(gaussian_max_shifts):
#         for pow_i, pow in enumerate(gaussian_powers):
#             y = get_gaussian_samples(n_samples, gaussian_dim, shift, pow)
#             for eps_i, epsilon in enumerate(n_epsilons):
#                 for rep in reps:
#                     r+=1
#                     print(f"\r{r/total_perfs}", end="")
#                     flow = model_dict[(gaussian_dim, epsilon, shift, pow, rep)]
#                     ll, transformed = flow.forward(y, marginalize=True)
#                     nll_bitsperdim = -((ll / 2)) / np.log(2)
#                     # Remove outliers
#                     perfs[dim_i,eps_i,shift_i,pow_i, rep] = reject_outliers(nll_bitsperdim.detach().numpy(),3).mean()
#
# pickle.dump(perfs, open("/home/guus/PycharmProjects/Thesis/Perfs/Gaussian_perfs","wb"))
perfs = pickle.load(open("/home/guus/PycharmProjects/Thesis/Perfs/Gaussian_perfs","rb"))
perfs_repav = perfs.mean(axis = -1)

dims,eps,shifts,pows,reps = perfs.shape


# Performance plots
# Group = dim
# fix: shift, pow
fig,ax = plt.subplots(1, pows, figsize = (15,5))
ax = ax.flatten()
for j, p in enumerate(gaussian_powers):
    ax[j].set_title(f"a={p}")
    for i,d in enumerate(n_gaussian_dims):
             ax[j].plot(n_epsilons, perfs_repav[i, :, 0, j], label = f"{d}-D")
ax[0].set_ylabel("-Ll (b/d)")
ax[0].set_xlabel("B")
plt.legend()
plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot1.png")
plt.show()


# Plot 2:
# group = shift
# fix: dim, pow
fig,ax = plt.subplots(1, shifts, figsize = (15,5))
ax = ax.flatten()
for j, p in enumerate(gaussian_max_shifts):
    ax[j].set_title(f"shift={p}")
    for i,d in enumerate(n_gaussian_dims):
             ax[j].plot(n_epsilons, perfs_repav[i, :,j , 1], label = f"{d}-D")
ax[0].set_ylabel("-Ll (b/d)")
ax[0].set_xlabel("B")
plt.legend()
plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot2.png")
plt.show()

# Plot 3:
# group = pow
# fix: shift, dim
fig,ax = plt.subplots(1, dims, figsize = (15,5))
ax = ax.flatten()
for j, p in enumerate(gaussian_max_shifts):
    ax[j].set_title(f"shift={p}")
    for i,d in enumerate(n_gaussian_dims):
             ax[j].plot(n_epsilons, perfs_repav[i, :,j,1], label = f"{d}-D")
ax[0].set_ylabel("-Ll (b/d)")
ax[0].set_xlabel("B")
plt.legend()
plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/Gaussian_plot3.png")
plt.show()