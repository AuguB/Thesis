import math

from Gym.trainer_bak import Trainer_bak
from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.Trainer import Trainer
import pickle
from time import *
import matplotlib.pyplot as plt
import numpy as np

class Baker:
    def __init__(self, n_samples = 2048, n_layers = 6, n_epochs = 1024, batch_size = 16, lr = 1e-3):
        self.n_samples = n_samples
        self.n_layers  = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        # Create a folder to store the test results


    def bake(self, dataname, n_epsilons,noise=None, clipNorm=None, make_plots =  False):
        self.dataname = dataname
        self.n_epsilons = n_epsilons
        self.current_test_folder = make_top_folder(f"{self.dataname}{self.n_layers}")
        pickle.dump(self.n_epsilons,open(f"{self.current_test_folder}/n_epsilons.p","wb"))
        start = time()
        rep_losses = []
        data = build_px_samples(self.dataname,self.n_samples)
        inputDim = int(np.prod(data.data.shape[1:]))
        for eps_i, epsilon in enumerate(self.n_epsilons):
            flow = marginalizingFlow(inputDim, epsilon, n_layers=self.n_layers)
            optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
            trainer = Trainer()
            losses = trainer.train(net=flow, dataset=data, optim=optim, n_epochs=self.n_epochs, batch_size=self.batch_size,
                                   dataname=self.dataname,noise=noise, clipNorm=clipNorm, make_plots=make_plots)
            rep_losses.append(losses)
            plt.plot(losses)
            checkpoint = {
                "optim": optim.state_dict(),
                "model": flow.state_dict()
            }
            torch.save(checkpoint, "/".join([self.current_test_folder, f"{self.dataname}_{self.n_layers}layers_{epsilon}eps_dict.p"]))
        losses_filename = "/".join([self.current_test_folder, "loss_list.p"])
        pickle.dump(rep_losses, open(losses_filename, "wb"))
        stop = time()
        duration = stop - start
        print(duration, " seconds")


    def bake_gaussian_models(self,n_epsilons):
        self.n_epsilons = n_epsilons
        n_gaussian_dims = [2, 3, 4]
        gaussian_powers = [1, 2, 3]
        n_repeats = 5
        start = time()
        # Create a folder to store the test results
        current_test_folder = make_top_folder(f"gaussian{self.n_layers}")

        # An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
        loss_dict = {}

        # An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
        model_dict = {}

        # Files which make it easy to find the losses and the models
        create_info_files(current_test_folder,
                          n_gaussian_dims,
                          self.n_epsilons,
                          n_repeats,
                          gaussian_powers)

        total_number_of_runs = len(n_gaussian_dims) \
                               * len(self.n_epsilons) \
                               * n_repeats \
                               * len(gaussian_powers)

        print(f"Going to train {total_number_of_runs} models")
        i = 0

        for dim_i, gaussian_dim in enumerate(n_gaussian_dims):
            for eps_i, epsilon in enumerate(self.n_epsilons):
                for pow_i, pow in enumerate(gaussian_powers):
                    for rep in range(n_repeats):
                        dataname = f"gauss_{gaussian_dim}dim_{epsilon}eps_{pow}pow_{rep}rep"
                        print(f"Now training model on {dataname}")
                        print(f"{round(100 * i / total_number_of_runs, 2)}% of all models trained   ")
                        i += 1
                        dataset = get_gaussian_samples(self.n_samples, gaussian_dim, pow)
                        flow = marginalizingFlow(N=gaussian_dim, M=epsilon, n_layers=self.n_layers)
                        optim = torch.optim.Adam(flow.parameters(), lr=self.lr)
                        trainer = Trainer()
                        losses = trainer.train(net=flow, dataset=dataset, optim=optim, n_epochs=self.n_epochs,
                                               batch_size=self.batch_size,
                                               dataname=dataname)


                        loss_dict[(gaussian_dim, epsilon, pow, rep)] = losses
                        print(len(losses))
                        plt.plot(losses)
                        plt.show()
                        checkpoint = {
                            "optim": optim.state_dict(),
                            "model": flow.state_dict()
                        }
                        torch.save(checkpoint, "/".join([current_test_folder,
                                                         f"{self.dataname}_{self.n_layers}_dict.p"]))

                        model_dict[(gaussian_dim, epsilon, pow, rep)] = flow.state_dict()

        # Plot the losses
        fig, ax = plt.subplots(1, 1)
        lookback = 16
        losses = [np.mean(losses[i:i + lookback]) for i in range(len(losses) - lookback)]
        ax.plot(losses)
        plt.title(f"{self.n_epochs}epochs {self.lr}lr")
        plt.show()

        models_filename = "/".join([current_test_folder, "model_dict.p"])
        pickle.dump(model_dict, open(models_filename, "wb"))
        losses_filename = "/".join([current_test_folder, "loss_dict.p"])
        pickle.dump(loss_dict, open(losses_filename, "wb"))

        stop = time()
        duration = stop - start
        print(duration, " seconds")
        print(f"The results are stored in {current_test_folder}")





