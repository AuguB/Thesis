import itertools
import pickle
import torch
from builder import build_px_samples, get_gaussian_samples
from config import project_folder
from marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from utils import moving_average

folder = "/home/guus/PycharmProjects/Thesis/Runs/run_2020-06-05 22:09:59.056985"
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

# Store the performances
n_samples = 300
perfs = np.zeros((len(n_gaussian_dims), len(n_epsilons), len(gaussian_max_shifts), len(gaussian_powers), len(reps)))
for dim_i, gaussian_dim in enumerate(n_gaussian_dims):
    for shift_i, shift in enumerate(gaussian_max_shifts):
        for pow_i, pow in enumerate(gaussian_powers):
            y = get_gaussian_samples(n_samples, gaussian_dim, shift, pow)
            for eps_i, epsilon in enumerate(n_epsilons):
                for rep in reps:
                    flow = model_dict[(gaussian_dim, epsilon, shift, pow, rep)]
                    log_prob_marg, transformed = flow.forward(y, marginalize=True)
                    perfs[dim_i, eps_i, shift_i,pow_i, rep] = -torch.mean(log_prob_marg).detach().numpy()


# Performance plots
# Plot 1: plot per power-dim, superimpose 1 line per eps average over reps
perfs_repav = perfs.mean(axis = -1)
dims,eps,shifts,pows,reps = perfs.shape
fig,ax = plt.subplots(pows,dims, figsize = (3*dims,3*pows))
for p in range(pows):
    for d in range(dims):
        ax[p,d].plot(n_epsilons,perfs_repav[d,:,0,p])


plt.show()

# fig, ax = plt.subplots(1, len(dists),figsize=(len(dists)*5,5))
# perfs = np.mean(perfs, axis = 2)
# for d, dist in enumerate(dists):
#     if len(dists) == 1:
#         ax.plot(epsilons, perfs[d])
#     else:
#         ax[d].plot(epsilons, perfs[d])
# plt.show()

# for d, dist in enumerate(dists):
#     for e, eps in enumerate(epsilons):
#         for r, rep in enumerate(reps):
#             flow = model_dict[dist,eps,r]
#             show_backward(flow,"")