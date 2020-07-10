import pickle
import torch
from builder import build_px_samples
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from plotters import plot_backward
from utils import moving_average, reject_outliers
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

folder = "/home/guus/PycharmProjects/Thesis/Runs/halfmoons4_2020-07-03 14:32:36.410025"
loss_dict = pickle.load(open(folder+"/loss_dict.p","rb"))
model_state_dict_dict = pickle.load(open(folder+"/model_dict.p","rb"))
n_repeats =5
n_epsilons = [0, 1, 2, 3,4]
model_dict = {}
logtwo = np.log(2)

# load the models
for eps in n_epsilons:
    for r in range(n_repeats):
        model_dict[(eps,r)] = marginalizingFlow(2, eps, n_layers = 6, mnist=True)
        model_dict[(eps,r)].load_state_dict(model_state_dict_dict[(eps,r)])
        model_dict[(eps,r)].eval()
print("models loaded")

# Plot the losses
for i,v in enumerate(n_epsilons):
    losses_lookback = 20
    for j in range(n_repeats):
        plt.plot(moving_average(loss_dict[v][j], losses_lookback))
        plt.show()
#
# Store the performances
n_samples = 200
# data = build_px_samples(n_samples,0,"half_moons")
# samples = data.data
# perfs = np.zeros((len(n_epsilons),n_samples, n_repeats))
# total = len(n_epsilons) * n_repeats
# for i,v in enumerate(n_epsilons):
#     for r in range(n_repeats):
#         ll,_ = model_dict[(v,r)](data, marginalize = True)
#         perfs[i, :, r] = -reject_outliers(ll.detach().numpy(),3)
#     pickle.dump(perfs, open("/home/guus/PycharmProjects/Thesis/Perfs/Halfmoons_perfs.p", "wb"))
perfs = pickle.load(open("/home/guus/PycharmProjects/Thesis/Perfs/Halfmoons_perfs.p","rb"))
perfs_samav = -np.log(np.exp(perfs).mean(axis=-2))
perfs_repav = np.log(np.exp(perfs_samav).mean(axis=-1))
perfs_minrep = np.min(perfs_samav, axis=-1)

print(" & ".join([str(round(i,2)) for i in perfs_minrep]))
#
#
# Performance plots
# Plot 1: one plot per repeat, and one average plt
fig,ax = plt.subplots(1,1, figsize = (4,4))
ci = 1.96 * np.abs(np.std(perfs_samav, axis=1) / np.sqrt(5))
ax.plot(n_epsilons, perfs_repav, c="black", alpha=1, lineStyle="-")
ax.fill_between(n_epsilons, perfs_repav + ci, perfs_repav - ci, color="black", alpha=0.1)
ax.set_ylabel("-Ll (b/d)")
ax.set_xlabel("B")
plt.tight_layout()
plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/Halfmoons_plot1.png')
plt.show()


fig,ax = plt.subplots(1, len(n_epsilons)+1, sharex=True, sharey=True, figsize = (15, 3))
for i, e  in enumerate(n_epsilons):
    model = model_dict[(e,0)]
    data = MultivariateNormal(loc=torch.zeros(model.Q), covariance_matrix=torch.diag(torch.ones(model.Q))).sample(
        (500,))
    inv = model.inverse(data).detach().numpy()[:,:2]
    print(inv.shape)
    ax[i].scatter(inv[:,0], inv[:,1], s = 1)
    ax[i].set_title( f"B={e}")
    ax[i].set_xlim((-2,3))
    ax[i].set_ylim((-1,1.5))
target = build_px_samples(500,0,"half_moons").detach().numpy()
ax[-1].scatter(target[:,0],target[:,1],s=1,c="red")
ax[-1].set_title("target")
ax[-1].set_xlim((-2, 3))
ax[-1].set_ylim((-1, 1.5))
plt.tight_layout()
plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/Halfmoons_plot2.png')
plt.show()


# for d, dist in enumerate(dists):
#     for e, eps in enumerate(epsilons):
#         for r, rep in enumerate(reps):
#             flow = model_dict[dist,eps,r]
#             show_backward(flow,"")