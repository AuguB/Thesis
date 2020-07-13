import pickle
import torch
from torch.distributions import MultivariateNormal

from Gym.Trainer import Trainer
from builder import build_px_samples, get_MNIST, get_CIFAR10
from Flows.marginalizingflow import marginalizingFlow
import numpy as np
import matplotlib.pyplot as plt

from plotters import plot_backward
from utils import moving_average, reject_outliers
from torch.utils.data import DataLoader

folder = "/home/guus/PycharmProjects/Thesis/Runs/cifar_checkpoints"
# losses = pickle.load(open(folder+"/loss_list.p","rb"))
# n_epsilons = pickle.load(open(folder+"/n_epsilons.p","rb"))
model_dict = {}
n_pixels = 3*32*32
logtwo = np.log(2)
#
eps = 0

checkpoint = torch.load(folder+f"/CIFAR_3layers_{eps}eps_dict.p")
net = marginalizingFlow(n_pixels, eps,n_layers = 3, mnist=True)
net.load_state_dict(checkpoint["model"])
net.eval()

data = MultivariateNormal(loc=torch.zeros(net.Q), covariance_matrix=torch.diag(torch.ones(net.Q))).sample((16,))
inv = net.inverse(data).detach().numpy()


fig,ax = plt.subplots(4,4,figsize=(16,16))
ax = ax.flatten()
for i in range(16):
    ax[i].imshow(np.swapaxes(np.swapaxes(inv[i].reshape((3,32,32)),0,2),0,1))
plt.savefig(f"{folder}/samples.png")
plt.show()

cont = True
if cont:
    lr = 5e-3
    n_epochs = 1
    batch = 64
    data = build_px_samples(100,0,"CIFAR10")

    trainer = Trainer()

    net = marginalizingFlow(n_pixels, eps, n_layers=3, mnist=True)
    net.load_state_dict(checkpoint["model"])
    optim = torch.optim.Adam(net.parameters(), lr)
    optim.load_state_dict(checkpoint["optim"])
    losses = trainer.train(net,data,optim,batch,dataname = "CIFAR10")

    data = MultivariateNormal(loc=torch.zeros(net.Q), covariance_matrix=torch.diag(torch.ones(net.Q))).sample((16,))
    inv = net.inverse(data).detach().numpy()


    fig,ax = plt.subplots(4,4,figsize=(16,16))
    ax = ax.flatten()
    for i in range(16):
        ax[i].imshow(np.swapaxes(np.swapaxes(inv[i].reshape((3,32,32)),0,2),0,1))
    plt.savefig(f"{folder}/samples.png")
    plt.show()

# # Plot the losses
# loss_dict = {}
# for i,v in enumerate(n_epsilons):
#     loss_dict[v] = losses[i]
#     losses_lookback = 20
#     plt.plot(moving_average(loss_dict[v], losses_lookback))
#     plt.show()
    # mnist_backward(model_dict[v], "")
#
#
# # Store the performances
# data = get_CIFAR10(False)
# shape = data.data.shape
# n_mnist_samples = shape[0]
# batchsize = 5000
# samples = data.data
# perfs = np.zeros((len(n_epsilons),n_mnist_samples))
# for i,v in enumerate(n_epsilons):
#     loader = DataLoader(dataset=data, batch_size=batchsize, shuffle=True)
#     for j, (d,_) in enumerate(loader):
#         d = torch.reshape(d, (-1,n_pixels))
#         d_noised = d + torch.rand(d.shape) * 0.2
#         ll,_ = model_dict[v](d_noised, marginalize = True, n_samples = 20)
#         print(ll.min(), ll.max(),ll.mean(),ll.std())
#         # print(ll)
#         perfs[i, j*batchsize:(j+1)*batchsize] = ll.detach().numpy()
#
# pickle.dump(perfs, open("/home/guus/PycharmProjects/Thesis/Perfs/CIFAR_logli.p", "wb"))
# perfs = pickle.load(open("/home/guus/PycharmProjects/Thesis/Perfs/CIFAR_logli.p", "rb")).astype(np.float128)
#
# perfs_samav = np.log(np.exp(perfs).mean(axis = -1))
#
# # #
# #  # Performance plots
# # # Plot 1: line plot, x = B, y = -ll (b/d)
# # fig,ax = plt.subplots(1,1, figsize = (4,4))
# # print("&".join([str(round(i,1)) for i in perfs_samav]), end = "\\\\\hline\n")
# # ax.plot(n_epsilons, perfs_samav)
# # ax.set_ylabel("Log likelihood")
# # ax.set_xlabel("B")
# # plt.tight_layout()
# # plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/MNIST_plot1.png')
# # plt.show()
# #
#
#
# # Performance plots
# # Plot 2: generations
# perfs_repav = perfs.mean(axis = -1)
# n_generations = 10
# fig,ax = plt.subplots(len(n_epsilons), n_generations, figsize = (20,14))
# plt.axis('off')
# for i,eps in enumerate(n_epsilons):
#     model = model_dict[eps]
#     p= MultivariateNormal(loc=torch.zeros(model.Q), covariance_matrix=torch.diag(torch.ones(model.Q)))
#     data = p.sample((n_generations,))
#     forward = model.inverse(data).detach().numpy()
#     for s in range(n_generations):
#         ax[i,s].imshow(forward[s,:784].reshape((28,28)), cmap = "Greys")
#         ax[i,s].set_xticks([])
#         ax[i,s].set_yticks([])
# plt.savefig('/home/guus/PycharmProjects/Thesis/Plots/MNIST_plot2.png')
# plt.show()
