import pickle
import torch
from builder import build_px_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from plotters import mnist_backward
from utils import moving_average, reject_outliers
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

folder = "/home/guus/PycharmProjects/Thesis/Runs/halfmoons6_2020-06-26 12:18:37.763193"
loss_dict = pickle.load(open(folder+"/loss_dict.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
n_repeats = 3
n_epsilons = [0, 1, 2, 4, 8, 16]
model_dict = {}
logtwo = np.log(2)

# load the models
for eps in n_epsilons:
    for r in range(n_repeats):
        model_dict[(eps,r)] = marginalizingFlow(2, eps, n_layers = 6, mnist=True)
        model_dict[(eps,r)].load_state_dict(model_state_dict_dict[(eps,r)])
        model_dict[(eps,r)].eval()
print("models loaded")
#
# # Plot the losses
# for i,v in enumerate(n_epsilons):
#     losses_lookback = 20
#     plt.plot(moving_average(loss_dict[v][0], losses_lookback))
#     plt.show()
# #
# # Store the performances
# n_samples = 300
# data = build_px_samples(n_samples,0,"half_moons")
# samples = data.data
# perfs = np.zeros((len(n_epsilons),n_samples, n_repeats))
# total = len(n_epsilons) * n_repeats
# for i,v in enumerate(n_epsilons):
#     for r in range(n_repeats):
#         ll,_ = model_dict[(v,r)](data, marginalize = True)
#         nll_bitsperdim = -((ll/2))/np.log(2)
#         perfs[i, :, r] = reject_outliers(nll_bitsperdim.detach().numpy(),3).copy()
# pickle.dump(perfs, open("/home/guus/PycharmProjects/Thesis/Perfs/Halfmoons_perfs.p", "wb"))
perfs = pickle.load(open("/home/guus/PycharmProjects/Thesis/Perfs/Halfmoons_perfs.p","rb"))
perfs_repav = perfs.mean(axis = -1)
perfs_av = perfs_repav.mean(axis = -1)
perfs_samav = perfs.mean(axis = -2)
#
#
# # Performance plots
# # Plot 1: one plot per repeat, and one average plt
# fig,ax = plt.subplots(1,1, figsize = (5,5))
# ax.plot(n_epsilons, perfs_av)
# for r in range(n_repeats):
#     ax.plot(n_epsilons,perfs_samav[:,r], alpha = .2, c = 'red')
# ax.set_title(f"Performance on the Half-moons dataset")
# ax.set_ylabel("-Ll (b/d)")
# ax.set_xlabel("B")
# plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/Halfmoons_plot1.png')
# plt.show()


fig,ax = plt.subplots( len(n_epsilons), n_repeats, sharex=True, sharey=True, figsize = (5, 10))
for i, e  in enumerate(n_epsilons):
    for r in range(n_repeats):
        model = model_dict[(e,r)]
        data = MultivariateNormal(loc=torch.zeros(model.Q), covariance_matrix=torch.diag(torch.ones(model.Q))).sample(
            (500,))
        inv = model.inverse(data).detach().numpy()[:,:2]
        print(inv.shape)
        ax[i, r].scatter(inv[:,0], inv[:,1], s = 1)
plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/Halfmoons_plot2.png')
plt.show()


# for d, dist in enumerate(dists):
#     for e, eps in enumerate(epsilons):
#         for r, rep in enumerate(reps):
#             flow = model_dict[dist,eps,r]
#             show_backward(flow,"")