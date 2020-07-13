import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

fat_alpha = 0.8
thin_alpha = 0.5
n_samples_to_plot = 500


def plot_losses(losses_storage, nth_distribution, distribution, n_epsilons, images_folder):
    slice_of_interest = losses_storage[nth_distribution]
    # average over repeats
    mean_over_repeats = np.mean(slice_of_interest,axis = 2)
    fig, ax = plt.subplots(1)
    for i,v in enumerate(mean_over_repeats):
        ax.plot(v, label = n_epsilons[i])
    plt.title(f"Train losses of {distribution}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.legend(title="Epsilon dimensions")
    plt.savefig("/".join([images_folder, "losses_"+distribution]))
    plt.show()
    plt.close()




def plot_performance(performance_storage, nth_distribution, distribution, n_epsilons, images_folder):
    slice_of_interest = performance_storage[nth_distribution]
    mean_over_repeats = np.mean(slice_of_interest, axis=2)
    fig, ax = plt.subplots(figsize = (7,7))
    ax.plot(n_epsilons, mean_over_repeats[:, 0])
    ax.xaxis.set_ticks(n_epsilons)
    ax.set_xlabel("auxilliary dimensions")
    ax.set_ylabel("Negative log likelihood")
    ax.set_title(f"Performance of model {distribution}", fontsize = 16)
    plt.savefig("/".join([images_folder, "performance_"+distribution]))
    plt.show()
    plt.close()
    pass


def show_forward(dataset, net, filename):
    with torch.no_grad():
        _, forward = net(dataset)
        forward = forward.detach().numpy()
        if forward.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(forward[:, 0], forward[:, 1], alpha=fat_alpha,s=1)
            ax.set_title("Forward")
            ax.set_ylim((-4, 4))
            ax.set_xlim((-4, 4))
        else:
            n_features = forward.shape[1]
            fig, ax = plt.subplots(n_features, n_features, figsize=(7, 7))
            for i in range(n_features):
                for j in range(n_features):
                    ax[i,j].scatter(forward[:, i], forward[:, j], alpha=thin_alpha, s=1)
            plt.title("Forward")
            plt.setp(ax, xlim = (-4,4), ylim = (-4,4))
        plt.show()
        # plt.savefig(filename)
        plt.close()


def show_backward(net, filename):
    data = MultivariateNormal(loc = torch.zeros(net.Q), covariance_matrix=torch.diag(torch.ones(net.Q))).sample((500,))
    with torch.no_grad():
        X = net.inverse(data).detach().numpy()
        if data.data.shape[1] == 2:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(X[:, 0], X[:, 1], alpha=fat_alpha,s=1)
            ax.set_title("Backward")
            ax.set_ylim((-4, 4))
            ax.set_xlim((-4, 4))
        else:
            n_features = X.shape[1]
            fig, ax = plt.subplots(n_features, n_features, figsize=(7, 7))
            for i in range(n_features):
                for j in range(n_features):
                    ax[i,j].scatter(X[:, i], X[:, j], alpha=thin_alpha, s=1)
            plt.setp(ax, xlim = (-4,4), ylim = (-4,4))
        plt.show()
        plt.close()

def plot_backward(net, dataname):
    if dataname == "MNIST":
        plot_mnist_backward(net)
    else:
        plot_cifar10_backward(net)

def plot_mnist_backward(net):
    data = MultivariateNormal(loc = torch.zeros(net.Q), covariance_matrix=torch.diag(torch.ones(net.Q))).sample((4,))
    with torch.no_grad():
        fig, ax = plt.subplots(2,2,figsize=(5, 5))
        ax = ax.flatten()
        backward = net.inverse(data).detach().numpy()[:,:784]
        for i in range(4):
            ax[i].imshow(np.reshape(backward[i],(28,28)), cmap='Greys')
        plt.show()
        plt.close(fig)

def plot_cifar10_backward(net):
    data = MultivariateNormal(loc=torch.zeros(net.Q), covariance_matrix=torch.diag(torch.ones(net.Q))).sample((4,))
    with torch.no_grad():

        backward = net.inverse(data).detach().numpy()[:, :3*32*32]
        backward = np.swapaxes(np.swapaxes(np.reshape(backward, (4, 3, 32, 32)) * 2 + 0.5, 1, 3), 1, 2)
        backward = np.clip(backward, 0, 1)
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        ax = ax.flatten()
        for i in range(4):
            ax[i].imshow(backward[i])
        plt.show()
        plt.close(fig)

def mnist_noised(noisy, natural):
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(np.reshape(natural.detach().numpy()[0,:784],(28,28)),cmap = 'Greys')
    ax[1].imshow(np.reshape(noisy.detach().numpy()[0,:784],(28,28)),cmap = 'Greys')

    plt.show()

def plot_mappings(flow, data, current_image_folder, modelname, dist):
    if dist == "MNIST":
        plotname = "/".join([current_image_folder, "plot_" + modelname])
        plot_backward(flow, plotname)
    else:
        forwardname = "/".join([current_image_folder, "forward_" + modelname])
        show_forward(data,flow,forwardname)
        backwardname = "/".join([current_image_folder, "backward_" + modelname])
        show_backward(flow, backwardname)


