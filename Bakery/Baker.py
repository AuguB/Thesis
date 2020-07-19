from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.Trainer import Trainer, train
import pickle
from time import *
import matplotlib.pyplot as plt
import numpy as np


class Baker:
    def __init__(self, device, n_samples=2048, n_layers=6, n_epochs=1024, batch_size=16, lr=1e-3, n_repeats=1):
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_repeats = [i for i in range(n_repeats)]
        self.device = device

    def bake(self, name_of_data, auxilliary_dimensons_list, noise=None, clip_norm=None, make_plots=False, print_status = False):
        self.name_of_data = name_of_data
        self.auxilliary_dimensons_list = auxilliary_dimensons_list
        self.folder = make_folder_for_data(f"{self.name_of_data}")
        self.make_and_store_param_dicts(clip_norm, noise)

        loss_dict = {}
        start_time = time()
        data = build_px_samples(self.name_of_data, self.n_samples)
        data_dimensions = int(np.prod(data.data.shape[1:]))
        for auxiliary_dimension_i, auxiliary_dimension in enumerate(self.auxilliary_dimensons_list):
            for repeat in self.n_repeats:
                repeat_start_time = time()
                print(f"Now going to train {self.name_of_data} with B={auxiliary_dimension}, repeat {repeat}")
                flow = marginalizingFlow(data_dimensions, auxiliary_dimension, n_layers=self.n_layers)
                optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
                losses = train(self.device, flow=flow, dataset=data, optim=optim, n_epochs=self.n_epochs,
                               batch_size=self.batch_size,
                               name_of_data=self.name_of_data, clip_norm=clip_norm, make_plots=make_plots, print_status = print_status)
                loss_dict[(auxiliary_dimension, repeat)] = losses
                checkpoint = {
                    "optim": optim.state_dict(),
                    "model": flow.state_dict()
                }
                print(f"Finished training {self.name_of_data} with B={auxiliary_dimension}, repeat {repeat} in {time()-repeat_start_time} seconds")

                torch.save(checkpoint, "/".join(
                    [self.folder, f"{self.name_of_data}_{auxiliary_dimension}eps_{repeat}rep.p"]))
        losses_filename = "/".join([self.folder, "loss_dict.p"])
        pickle.dump(loss_dict, open(losses_filename, "wb"))
        stop_time = time()
        duration = stop_time - start_time
        print(duration, " seconds")
        print(f"The results are stored in {self.folder}")
        return self.folder

    def bake_gaussian_models(self, auxilliary_dimensons_list, gaussian_dims, gaussian_exponents, make_plots=False):
        self.auxilliary_dimensons_list = auxilliary_dimensons_list
        start_time = time()
        # Create a folder to store the test results
        self.folder = make_folder_for_data(f"gaussian")

        # An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
        loss_dict = {}

        # An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
        model_dict = {}

        self.make_and_store_gaussian_param_dicts(auxilliary_dimensons_list, gaussian_dims, gaussian_exponents)

        for dim_i, gaussian_dim in enumerate(gaussian_dims):
            for eps_i, epsilon in enumerate(self.auxilliary_dimensons_list):
                for pow_i, pow in enumerate(gaussian_exponents):
                    for rep in self.n_repeats:
                        name_of_data = f"GAUSS_{gaussian_dim}dim_{pow}pow_{epsilon}eps_{rep}rep"
                        print(f"Now going to train {name_of_data}")
                        dataset = get_gaussian_samples(self.n_samples, gaussian_dim, pow)
                        flow = marginalizingFlow(data_dimensions=gaussian_dim, auxiliary_dimensions=epsilon,
                                                 n_layers=self.n_layers)
                        optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
                        losses = train(self.device, flow=flow, dataset=dataset, optim=optim,
                                       n_epochs=self.n_epochs,
                                       batch_size=self.batch_size,
                                       name_of_data=name_of_data,
                                       make_plots=make_plots)

                        loss_dict[(gaussian_dim, pow, epsilon, rep)] = losses
                        plt.plot(losses)
                        plt.show()
                        checkpoint = {
                            "optim": optim.state_dict(),
                            "model": flow.state_dict()
                        }
                        print(f"Finished training {name_of_data}")
                        torch.save(checkpoint, "/".join([self.folder,
                                                         f"{name_of_data}.p"]))

        # Plot the losses
        fig, ax = plt.subplots(1, 1)
        losses_lookback = 16
        losses = [np.mean(losses[i:i + losses_lookback]) for i in range(len(losses) - losses_lookback)]
        ax.plot(losses)

        models_filename = "/".join([self.folder, "model_dict.p"])
        pickle.dump(model_dict, open(models_filename, "wb"))
        losses_filename = "/".join([self.folder, "loss_dict.p"])
        pickle.dump(loss_dict, open(losses_filename, "wb"))

        stop = time()
        duration = stop - start_time
        print(duration, " seconds")
        print(f"The results are stored in {self.folder}")

    def make_and_store_param_dicts(self, clip_norm, noise):
        model_param_dict = \
            {
                "n_layers": self.n_layers,
                "dataname": self.name_of_data,
                "noise": noise,
                "clipNorm": clip_norm
            }
        pickle.dump(model_param_dict, open(f"{self.folder}/info_dict.p", "wb"))  # was info_dict.p
        train_param_dict = \
            {
                "n_epsilons": self.auxilliary_dimensons_list,
                "n_repeats": self.n_repeats,
            }
        pickle.dump(train_param_dict, open(f"{self.folder}/param_dict.p", "wb"))  # was param_dict.p

    def make_and_store_gaussian_param_dicts(self, auxilliary_dimensons_list, gaussian_dims, gaussian_exponents):
        model_param_dict = \
            {
                "n_layers": self.n_layers,
                "dataname": "GAUSS"
            }
        pickle.dump(model_param_dict, open(f"{self.folder}/info_dict.p", "wb"))
        train_param_dict = {
            "n_gaussian_dims": gaussian_dims,
            "gaussian_powers": gaussian_exponents,
            "n_epsilons": auxilliary_dimensons_list,
            "n_repeats": self.n_repeats
        }
        pickle.dump(train_param_dict, open(f"{self.folder}/param_dict.p", "wb"))  # Was param_dict.p
