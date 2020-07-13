import pickle
from datetime import *
import os
from sklearn import datasets
from torchvision import datasets as tds
from torchvision.transforms import transforms

from Gym.My_Dataset import MyDataset
from config import project_folder, runs_folder
from Flows.flows import *
from Flows.normalizingflow import *


def build_flow(dim, n_layers=3):
    module_list = []
    module_list.append(actnorm(dim))
    for l in range(n_layers):
        module_list.append(coupling(dim))
        module_list.append(shuffle(dim,True))
        module_list.append(coupling(dim))
        module_list.append(shuffle(dim,False))
        module_list.append(actnorm(dim))
    module_list.pop()
    module_list.pop()
    return normalizingFlow(module_list, dim)


def build_px_samples(distribution="square_gaussian", n_samples=2048):
    if distribution == "square_gaussian":
        base = torch.randn((n_samples, 1))
        return MyDataset(base**2)

    elif distribution == "half_moons":
        data, _ = datasets.make_moons(n_samples, shuffle=True, noise=0.1)
        return MyDataset(toTorch(data))

    elif distribution == "blobs":
        data,_= datasets.make_blobs(n_samples, centers=[[-4, 4], [-4, -4], [4, 4], [4, -4]], shuffle=True)
        return MyDataset(toTorch(data))

    elif distribution == "MNIST":
        return get_MNIST()

    elif distribution == "CIFAR10":
        return get_CIFAR10()

    else:
        print("Distribution unknown")
        return None


def get_MNIST(train = True):
    dataset = tds.MNIST('../data', train=train, download=True)
    transform =transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ])
    return MyDataset(dataset.data.numpy(), transform)

def get_CIFAR10(train=True):
    dataset = tds.CIFAR10(root='../data',train=train,download=True)
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
    return MyDataset(dataset.data, transform)

def get_gaussian_samples(n_samples,gaussian_dim, pow):
    dist = torch.distributions.MultivariateNormal(loc=torch.zeros((gaussian_dim)), covariance_matrix=torch.diag(torch.ones((gaussian_dim))))
    x = dist.sample((n_samples,))**pow
    return MyDataset(x)



def toTorch(A):
    return torch.from_numpy(A).type("torch.FloatTensor")

def make_top_folder(name=""):
    current_test_folder = "/".join([project_folder, runs_folder,
                                    f"{name}_{datetime.now()}"])
    os.mkdir(current_test_folder)
    return current_test_folder

def create_info_files(main_folder, n_gaussian_dims,
                      n_epsilons,
                      n_repeats,
                      gaussian_powers):
    filename = "/".join([main_folder, "param_dict.p"])
    dict = {
        "n_gaussian_dims": n_gaussian_dims,
        "n_epsilons": n_epsilons,
        "n_repeats": n_repeats,
        "gaussian_powers": gaussian_powers}
    pickle.dump(dict, open(filename, "wb"))