import pickle
import torch
from builder import build_px_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from plotters import mnist_backward
from utils import moving_average
from torch.utils.data import DataLoader

folder = "/home/guus/PycharmProjects/Thesis/Runs/MNIST4_2020-06-26 15:23:09.058579"
losses = pickle.load(open(folder+"/loss_list.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
n_epsilons = pickle.load(open(folder+"/n_epsilons.p","rb"))
model_dict = {}
n_pixels = 784
logtwo = np.log(2)

# load the models
for eps in n_epsilons:
    model_dict[eps] = marginalizingFlow(784, eps,n_layers = 4, mnist=True)
    model_dict[eps].load_state_dict(model_state_dict_dict[eps])
    model_dict[eps].eval()

# Plot the losses
loss_dict = {}
for i,v in enumerate(n_epsilons):
    loss_dict[v] = losses[i]
    losses_lookback = 20
    plt.plot(moving_average(loss_dict[v], losses_lookback))
    plt.show()
    # mnist_backward(model_dict[v], "")


# Store the performances
data = build_px_samples(1,0,"MNIST")
n_mnist_samples = 60000
samples = data.data
perfs = np.zeros((len(n_epsilons),n_mnist_samples))
for i,v in enumerate(n_epsilons):
    loader = DataLoader(dataset=data, batch_size=100, shuffle=True)
    for j, (d,_) in enumerate(loader):
        d = torch.reshape(d, (-1,n_pixels))
        d_noised = d + torch.rand(d.shape) * 0.2
        ll,_ = model_dict[v](d_noised, marginalize = True)
        # print(ll)
        nll_bitsperdim = -((ll/n_pixels)-np.log(128))/np.log(2)
        perfs[i, j*100:(j+1)*100] = nll_bitsperdim

pickle.dump(perfs, open("/home/guus/PycharmProjects/Thesis/Perfs/MNIST_perfs.p", "wb"))
perfs = pickle.load(open("/home/guus/PycharmProjects/Thesis/Perfs/MNIST_perfs.p","rb"))
perfs_samav = perfs.mean(axis = -1)
 # Performance plots
# Plot 1: one plot per repeat, and one average plt
fig,ax = plt.subplots(1,1, figsize = (5,5))
ax.plot(n_epsilons, perfs_samav)

ax.set_title(f"Performance on the MNIST dataset")
ax.set_ylabel("-Ll (b/d)")
ax.set_xlabel("B")
plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/MNIST_plot1.png')
plt.show()


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