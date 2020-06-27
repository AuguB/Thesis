import pickle
import torch
from builder import build_px_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from plotters import mnist_backward
from utils import moving_average, reject_outliers
from torch.utils.data import DataLoader

folder = "/home/guus/PycharmProjects/Thesis/Runs/halfmoons6_2020-06-26 12:18:37.763193"
losses = pickle.load(open(folder+"/loss_list.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
n_epsilons = pickle.load(open(folder+"/n_epsilons.p","rb"))
model_dict = {}
logtwo = np.log(2)

# load the models
for eps in n_epsilons:
    model_dict[eps] = marginalizingFlow(2, eps, n_layers = 6, mnist=True)
    model_dict[eps].load_state_dict(model_state_dict_dict[eps])
    model_dict[eps].eval()

# Plot the losses
loss_dict = {}
for i,v in enumerate(n_epsilons):
    loss_dict[v] = losses[i]
    losses_lookback = 20
    plt.plot(moving_average(loss_dict[v], losses_lookback))
    plt.show()

# Store the performances
n_samples = 2000
data = build_px_samples(n_samples,0,"half_moons")
samples = data.data
perfs = np.zeros((len(n_epsilons),n_samples))
for i,v in enumerate(n_epsilons):
    loader = DataLoader(dataset=data, batch_size=100, shuffle=True)
    for j, d in enumerate(loader):
        ll,_ = model_dict[v](d, marginalize = True)
        nll_bitsperdim = -((ll/2)-np.log(128))/np.log(2)
        perfs[i, j*100:(j+1)*100] = reject_outliers(nll_bitsperdim.detach().numpy(),3).mean()


#
# # Performance plots
# # Plot 1: plot per power-dim, superimpose 1 line per eps average over reps
# perfs_repav = perfs.mean(axis = -1)
# dims,eps,shifts,pows,reps = perfs.shape
# fig,ax = plt.subplots(pows,dims, figsize = (3*dims,3*pows))
# for p in range(pows):
#     for d in range(dims):
#         # ax[p, d].plot(n_epsilons, perfs_repav[d, :, 1, p])
#         # ax[p, d].set_title(f"{n_gaussian_dims[d]}D Gaussian with power {gaussian_powers[p]}")
#
#         for s in range(shifts):
#             ax[p,d].set_title(f"{n_gaussian_dims[d]}d Gaussian with power {gaussian_powers[p]}")
#             ax[p,d].plot(n_epsilons,perfs_repav[d,:,s,p], label = f"{gaussian_max_shifts[s]}")
#
# plt.suptitle("Some results" )
# plt.show()

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