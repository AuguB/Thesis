import math

from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.Trainer import Trainer
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
        # Create a folder to store the test results

    def bake(self, dataname, n_epsilons, noise=None, clipNorm=None, make_plots=False):
        self.dataname = dataname
        self.n_epsilons = n_epsilons
        self.current_test_folder = make_top_folder(f"{self.dataname}")

        info_dict = \
            {
                "n_layers": self.n_layers,
                "dataname": self.dataname,
                "noise": noise,
                "clipNorm": clipNorm
            }
        pickle.dump(info_dict, open(f"{self.current_test_folder}/info_dict.p", "wb"))

        param_dict = \
            {
                "n_epsilons": self.n_epsilons,
                "n_repeats": self.n_repeats,
            }
        pickle.dump(param_dict, open(f"{self.current_test_folder}/param_dict.p", "wb"))

        loss_dict = {}
        start = time()
        data = build_px_samples(self.dataname, self.n_samples)
        inputDim = int(np.prod(data.data.shape[1:]))
        for eps_i, epsilon in enumerate(self.n_epsilons):
            for r in self.n_repeats:
                print(f"Now going to train {self.dataname} with B={epsilon}, repeat {r}")
                flow = marginalizingFlow(inputDim, epsilon, n_layers=self.n_layers)
                optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
                trainer = Trainer()
                losses = trainer.train(self.device, net=flow, dataset=data, optim=optim, n_epochs=self.n_epochs,
                                       batch_size=self.batch_size,
                                       dataname=self.dataname, clipNorm=clipNorm, make_plots=make_plots)
                loss_dict[(epsilon, r)] = losses
                checkpoint = {
                    "optim": optim.state_dict(),
                    "model": flow.state_dict()
                }
                print(f"Finished training {self.dataname} with B={epsilon}, repeat {r}")

                torch.save(checkpoint, "/".join(
                    [self.current_test_folder, f"{self.dataname}_{epsilon}eps_{r}rep.p"]))
        losses_filename = "/".join([self.current_test_folder, "loss_dict.p"])
        pickle.dump(loss_dict, open(losses_filename, "wb"))
        stop = time()
        duration = stop - start
        print(duration, " seconds")
        print(f"The results are stored in {self.current_test_folder}")


    def bake_gaussian_models(self, eps, dims, pows, make_plots=False):
        self.n_epsilons = eps
        start = time()
        # Create a folder to store the test results
        self.current_test_folder = make_top_folder(f"gaussian")

        # An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
        loss_dict = {}

        # An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
        model_dict = {}

        # Files which make it easy to find the losses and the models
        create_param_dict(self.current_test_folder,
                          dims,
                          pows,
                          self.n_epsilons,
                          self.n_repeats
                          )

        info_dict = \
            {
                "n_layers": self.n_layers,
                "dataname": "GAUSS"
            }
        pickle.dump(info_dict, open(f"{self.current_test_folder}/info_dict.p", "wb"))

        total_number_of_runs = len(dims) \
                               * len(self.n_epsilons) \
                               * len(self.n_repeats) \
                               * len(pows)


        for dim_i, gaussian_dim in enumerate(dims):
            for eps_i, epsilon in enumerate(self.n_epsilons):
                for pow_i, pow in enumerate(pows):
                    for rep in self.n_repeats:
                        dataname = f"GAUSS_{gaussian_dim}dim_{pow}pow_{epsilon}eps_{rep}rep"
                        print(f"Now going to train {dataname}")
                        dataset = get_gaussian_samples(self.n_samples, gaussian_dim, pow)
                        flow = marginalizingFlow(N=gaussian_dim, M=epsilon, n_layers=self.n_layers)
                        optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
                        trainer = Trainer()
                        losses = trainer.train(self.device,net=flow, dataset=dataset, optim=optim, n_epochs=self.n_epochs,
                                               batch_size=self.batch_size,
                                               dataname=dataname,
                                               make_plots=make_plots)

                        loss_dict[(gaussian_dim, pow, epsilon, rep)] = losses
                        plt.plot(losses)
                        plt.show()
                        checkpoint = {
                            "optim": optim.state_dict(),
                            "model": flow.state_dict()
                        }
                        print(f"Finished training {dataname}")
                        torch.save(checkpoint, "/".join([self.current_test_folder,
                                                         f"{dataname}.p"]))

        # Plot the losses
        fig, ax = plt.subplots(1, 1)
        lookback = 16
        losses = [np.mean(losses[i:i + lookback]) for i in range(len(losses) - lookback)]
        ax.plot(losses)

        models_filename = "/".join([self.current_test_folder, "model_dict.p"])
        pickle.dump(model_dict, open(models_filename, "wb"))
        losses_filename = "/".join([self.current_test_folder, "loss_dict.p"])
        pickle.dump(loss_dict, open(losses_filename, "wb"))

        stop = time()
        duration = stop - start
        print(duration, " seconds")
        print(f"The results are stored in {self.current_test_folder}")


