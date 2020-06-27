import pickle
from datetime import *
import os
from sklearn import datasets
from torchvision import datasets as tds
from torchvision.transforms import transforms

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

def build_mnist_flow(dim, n_layers=3):
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

def build_px_samples(n_samples, n_epsilons=0, distribution="square_gaussian"):
    if distribution == "square_gaussian":
        base = torch.randn((n_samples, 1)) ** 2
    elif distribution == "half_moons":
        base, _ = datasets.make_moons(n_samples, shuffle=True, noise=0.1)
        base = torch.from_numpy(base).type("torch.FloatTensor")
    elif distribution == "blobs":
        base, _ = datasets.make_blobs(n_samples, centers=[[-4, 4], [-4, -4], [4, 4], [4, -4]], shuffle=True)
        base = torch.from_numpy(base).type("torch.FloatTensor")
    elif distribution == "MNIST":
        base = get_MNIST()
        n_samples = base.data.shape[0]
    else:
        print("Distribution unknown")
        return None
    # If we need epsilon, add it
    if n_epsilons > 0:
        epsilon = torch.randn((n_samples, n_epsilons))
        result = torch.cat([base, epsilon], dim=1)
    # otherwise, we are good
    else:
        result = base
    return result


def make_top_folder(name=""):
    current_test_folder = "/".join([project_folder, runs_folder,
                                    f"{name}_{datetime.now()}"])
    os.mkdir(current_test_folder)
    return current_test_folder


def current_folders(images_folder, models_folder, p, shape_x):
    model_signature = f"{shape_x}_{p}pad"
    current_image_folder = "/".join([images_folder, model_signature])
    os.mkdir(current_image_folder)
    current_models_folder = "/".join([models_folder, model_signature])
    os.mkdir(current_models_folder)
    return current_image_folder, current_models_folder, model_signature


def distribution_folders(current_test_folder, shape_x):
    super_folder = "/".join([current_test_folder, shape_x])
    os.mkdir(super_folder)
    images_folder = "/".join([super_folder, "images"])
    models_folder = "/".join([super_folder, "models"])
    os.mkdir(images_folder)
    os.mkdir(models_folder)
    return images_folder, models_folder


def create_info_files(main_folder, n_gaussian_dims,
                      n_epsilons,
                      n_repeats,
                      gaussian_max_shifts,
                      gaussian_powers):
    filename = "/".join([main_folder, "param_dict.p"])
    dict = {
        "n_gaussian_dims": n_gaussian_dims,
        "n_epsilons": n_epsilons,
        "n_repeats": n_repeats,
        "gaussian_max_shifts": gaussian_max_shifts,
        "gaussian_powers": gaussian_powers}
    pickle.dump(dict, open(filename, "wb"))

def get_gaussian_samples(n_samples,gaussian_dim,shift, pow):
    dist = torch.distributions.MultivariateNormal(loc=torch.ones((gaussian_dim))*shift, covariance_matrix=torch.diag(torch.ones((gaussian_dim))))
    x = dist.sample((n_samples,))**pow
    return x


def get_MNIST():
    dataset = tds.MNIST('../data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    return dataset