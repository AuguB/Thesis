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
from utils import AddGaussianNoise, Flattener, timecode


def build_flow(dim, n_layers=3, device = None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module_list = []
    module_list.append(actnorm(dim))
    for l in range(n_layers):
        module_list.append(coupling(dim))
        module_list.append(shuffle(dim, True))
        module_list.append(coupling(dim))
        module_list.append(shuffle(dim, False))
        module_list.append(actnorm(dim))
    module_list.pop()
    module_list.pop()
    return normalizingFlow(module_list, dim, device)


def build_px_samples(distribution, gauss_params=None, n_samples=2048):
    if distribution == "HALFMOONS":
        data, _ = datasets.make_moons(n_samples, shuffle=True, noise=0.1)
        return MyDataset(toTorch(data))

    elif distribution == "MNIST":
        return get_MNIST()

    elif distribution == "FMNIST":
        return get_FMNIST()

    elif distribution == "KMNIST":
        return get_KMNIST()

    elif distribution == "CIFAR10":
        return get_CIFAR10()

    elif distribution == "GAUSS":
        return get_gaussian_samples(n_samples, gauss_params[0], gauss_params[1])

    else:
        print("Distribution unknown")
        return None


def get_MNIST(train=True):
    dataset = tds.MNIST('../data', train=train, download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0, 0.2),
        Flattener()
    ])
    return MyDataset(dataset.data.numpy(), transform)


def get_KMNIST(train=True):
    dataset = tds.KMNIST('../data', train=train, download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(0, 0.2),
        Flattener()
    ])
    return MyDataset(dataset.data.numpy(), transform)


def get_FMNIST(train=True):
    dataset = tds.FashionMNIST('../data', train=train, download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        AddGaussianNoise(0, 0.2),
        Flattener()
    ])
    return MyDataset(dataset.data.numpy(), transform)


def get_CIFAR10(train=True):
    dataset = tds.CIFAR10(root='../data', train=train, download=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Flattener()
    ])
    return MyDataset(dataset.data, transform)


def get_gaussian_samples(n_samples, gaussian_dim, pow):
    dist = torch.distributions.MultivariateNormal(loc=torch.zeros((gaussian_dim)),
                                                  covariance_matrix=torch.diag(torch.ones((gaussian_dim))))
    x = dist.sample((n_samples,)) ** pow
    return MyDataset(x)


def toTorch(A):
    return torch.from_numpy(A).type("torch.FloatTensor")


def make_top_folder(name=""):
    current_test_folder = "/".join([project_folder, runs_folder,
                                    f"{name}_{timecode()}"])
    os.mkdir(current_test_folder)
    return current_test_folder


def create_param_dict(main_folder, n_gaussian_dims,
                      gaussian_powers,
                      n_epsilons,
                      n_repeats
                      ):
    filename = "/".join([main_folder, "param_dict.p"])
    dict = {
        "n_gaussian_dims": n_gaussian_dims,
        "gaussian_powers": gaussian_powers,
        "n_epsilons": n_epsilons,
        "n_repeats": n_repeats
    }
    pickle.dump(dict, open(filename, "wb"))
