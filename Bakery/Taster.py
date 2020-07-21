import itertools
import math
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from Flows.marginalizingflow import marginalizingFlow
from builder import get_gaussian_samples, build_px_samples
from plotters import plot_and_store_backward_pass
from utils import moving_average


def print_summary(iter, total_iter, conf, total_confs):
    print(f"\rNow evaluating model {conf}/{total_confs}, {100*iter / total_iter}%", end="")


class Taster:
    def __init__(self, device, folder):
        self.device = device
        self.folder = folder
        self.loss_dict = pickle.load(open(folder + "/loss_dict.p", "rb"))
        self.model_param_dict = pickle.load(open(folder + "/info_dict.p", "rb"))
        self.train_param_dict = pickle.load(open(folder + "/param_dict.p", "rb"))
        self.taste_gaussian_models = self.model_param_dict["dataname"] == "GAUSS"
        self.logli = None
        self.logli_average_over_samples = None
        self.logli_average_over_repeats = None
        if self.taste_gaussian_models:
            self.model_state_dict_dict = pickle.load(open(folder + "/model_dict.p", "rb"))

    def plot_losses(self):
        for param_tuple in self.loss_dict.keys():
            if self.taste_gaussian_models:
                name_of_model = f"GAUSS_{param_tuple[0]}dim_{param_tuple[1]}pow_{param_tuple[2]}eps_{param_tuple[3]}rep"
            else:
                name_of_data = self.model_param_dict["dataname"]
                name_of_model = f"{name_of_data}_{param_tuple[0]}eps_{param_tuple[1]}rep"
            losses_of_model = self.loss_dict[param_tuple]
            losses_lookback = 5
            plt.plot(moving_average(losses_of_model, losses_lookback))
            plt.title(name_of_model)
            plt.show()

    def compute_logli(self, precomputed=False):
        if not precomputed:
            parameter_configurations_with_indices = itertools.product(
                *[zip([param_index for param_index in range(len(param_list))], param_list) for param_list in
                  self.train_param_dict.values()])

            n_data_points, n_samples_from_epsilon = self.get_sample_numbers()
            logli_buffer = np.zeros([len(i) for i in self.train_param_dict.values()] + [n_data_points])

            total_configs = np.product([len(i) for i in self.train_param_dict.values()])
            for parameters_i, parameters in enumerate(parameter_configurations_with_indices):
                if self.taste_gaussian_models:
                    print(f"\rNow testing model {parameters_i} of {total_configs}    ", end="")
                    n_samples_from_epsilon = 200
                    n_data_samples = 2048
                    ((dim_i, dim), (pow_i, pow), (epsilon_i, epsilon), (rep_i, rep)) = parameters
                    model = self.get_model(parameters)
                    data = get_gaussian_samples(n_data_samples, dim,
                                                pow).data.to(self.device)  # Because the data is low-dim, we can do a forward pass on
                    # the whole set at once, and we don't need dataloader.
                    # That's why here the data is taken directly from the dataset.
                    logli, _ = model(data.data, marginalize=True, n_samples=n_samples_from_epsilon)
                    logli_buffer[dim_i, pow_i, epsilon_i, rep_i, :logli.shape[0]] = logli.detach().cpu().numpy()

                else:
                    # Unpack tuple with params and indices
                    print(f"\rNow testing model {parameters_i} of {total_configs}    ", end="")

                    ((epsilon_i, epsilon), (repeat_i, repeat)) = parameters

                    data = build_px_samples(self.model_param_dict["dataname"])
                    data.data = data.data[:n_data_points]
                    batch_size = 250
                    loader = DataLoader(data, batch_size=batch_size)
                    total_iter = math.ceil(len(data) / batch_size)

                    # Get model
                    model = self.get_model(parameters)

                    # Compute performance by batch
                    for batch_i, batch in enumerate(loader):
                        batch = batch.to(self.device)
                        print_summary(batch_i, total_iter, parameters_i,total_configs)
                        current_logli, _ = model(batch, marginalize=True, n_samples=n_samples_from_epsilon)
                        logli_buffer[epsilon_i, repeat_i,
                        batch_i * batch_size:(batch_i + 1) * batch_size] = current_logli.detach().cpu().numpy().copy()

            pickle.dump(logli_buffer, open(self.folder + "/logli_buffer.p", "wb"))

        self.logli = pickle.load(open(self.folder + "/logli_buffer.p", "rb"))
        self.logli_average_over_samples = np.log(np.exp(self.logli).mean(axis=-1))
        self.logli_average_over_repeats = np.log(np.exp(self.logli_average_over_samples).mean(axis=-1))

    def get_model(self, parameters):
        if self.taste_gaussian_models:
            ((i, dim), (j, pow), (k, eps), (l, rep)) = parameters
            name_of_model = f"GAUSS_{dim}dim_{pow}pow_{eps}eps_{rep}rep"
            checkpoint = torch.load(self.folder + "/" + name_of_model + ".p", map_location=self.device)
            model = marginalizingFlow(dim, eps, self.model_param_dict["n_layers"])
        else:
            ((i, eps), (j, reps)) = parameters
            name = self.model_param_dict["dataname"]
            name_of_model = f"{name}_{eps}eps_{reps}rep"
            checkpoint = torch.load(self.folder + "/" + name_of_model + ".p", map_location=self.device)
            model = marginalizingFlow(self.get_dim(name), eps, self.model_param_dict["n_layers"])
        model.load_state_dict(checkpoint["model"])
        model.eval()
        model = model.to(self.device)
        return model

    def plot_avg_logli(self):
        if not self.taste_gaussian_models:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(self.train_param_dict["n_epsilons"], self.logli_average_over_repeats)
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            ax.set_xticks(self.train_param_dict["n_epsilons"])
            if len(self.train_param_dict["n_repeats"]) > 1:  # plot confidence intervals
                ci = 1.96 * np.abs(np.std(self.logli_average_over_samples, axis=1) / np.sqrt(5))
                ax.fill_between(self.train_param_dict["n_epsilons"], self.logli_average_over_repeats + ci,
                                self.logli_average_over_repeats - ci, alpha=0.1)
            name = self.model_param_dict["dataname"]

            plt.savefig(self.folder + f"/{name}_average_logli.png")
            plt.show()
        else:
            line_styles = ["-", "--", "-."]
            colors = ["red", "blue", "green"]
            for j, p in enumerate(self.train_param_dict["gaussian_powers"]):
                print(p)
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={p}")
                for i, d in enumerate(self.train_param_dict["n_gaussian_dims"]):
                    current_range_average_over_repeats = self.logli_average_over_repeats[i, j]
                    current_range_average_over_samples = self.logli_average_over_samples[i, j]
                    ci = 1.96 * np.abs(np.std(current_range_average_over_samples, axis=-1) / np.sqrt(5))
                    ax.plot(self.train_param_dict["n_epsilons"], current_range_average_over_repeats, label=f"{d}-D",
                            c=colors[i], alpha=1,
                            lineStyle=line_styles[i])
                    if len(self.train_param_dict["n_repeats"]) > 1:
                        ax.fill_between(self.train_param_dict["n_epsilons"], current_range_average_over_repeats + ci,
                                        current_range_average_over_repeats - ci,
                                        color=colors[i], alpha=0.1)
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if j == 0:
                    plt.legend()
                plt.tight_layout()
                name = self.model_param_dict["dataname"]

                plt.savefig(self.folder + f"/{name}average_logli{p}.png")
                plt.show()

    def plot_max_logli(self):
        if not self.taste_gaussian_models:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(self.train_param_dict["n_epsilons"], np.max(self.logli_average_over_samples, axis=-1),c="black")
            ax.set_xlabel("B")
            ax.set_ylabel("Log Likelihood")
            ax.set_xticks(self.train_param_dict["n_epsilons"])
            plt.tight_layout()
            name = self.model_param_dict["dataname"]

            plt.savefig(self.folder + f"/{name}_max_logli.png")
            plt.show()
        else:
            line_styles = ["-", "--", "-."]
            colors = ["black", "black", "black"]
            for power_i, power in enumerate(self.train_param_dict["gaussian_powers"]):
                fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
                ax.set_title(f"ψ={power}")
                for dim_i, dim in enumerate(self.train_param_dict["n_gaussian_dims"]):
                    current_range_average_over_samples = self.logli_average_over_samples[dim_i, power_i]
                    ax.plot(self.train_param_dict["n_epsilons"], np.max(current_range_average_over_samples, axis=-1),
                            label=f"{dim}-D",
                            c=colors[dim_i], alpha=1,
                            lineStyle=line_styles[dim_i])
                ax.set_xlabel("B")
                ax.set_ylabel("Log Likelihood")
                if power_i == 0:
                    plt.legend()
                # plt.tight_layout()
                plt.xticks([str(i) for i in self.train_param_dict["n_epsilons"]])
                name = self.model_param_dict["dataname"]
                plt.savefig(self.folder + f"/{name}_max_logli{power}.png")
                plt.show()

    def generate(self, plot_best_repeats = True):
        if self.taste_gaussian_models:
            self.generate_gaussian_forward_plot_matrix(plot_best_repeats)
        elif self.model_param_dict["dataname"] == "HALFMOONS":
            self.generate_halfmoons_forward_plot_matrix(plot_best_repeats)
        else:
            self.generate_image_forward_plot_matrix(self.model_param_dict["dataname"])

    def get_importance_samples(self, a):
        if a.endswith("MNIST") or a.endswith("CIFAR10"):
            return 10
        else:
            return 200

    def get_dim(self, a):
        if a.endswith("MNIST"):
            return 28 * 28
        elif a.endswith("CIFAR10"):
            return 3 * 32 * 32
        elif a == "HALFMOONS":
            return 2
        else:
            return 2

    def get_sample_numbers(self):
        data_name: str = self.model_param_dict["dataname"]
        if self.taste_gaussian_models:
            return 2048, 200
        elif data_name == "HALFMOONS":
            return 2048, 200
        elif data_name.endswith("MNIST"):
            return 60000, 100
        else:
            return 50000,100

    def print_best_model_table(self):
        if self.taste_gaussian_models:
            print("\\psi&\tD&\tB=" + "&\t B=".join([str(i) for i in self.train_param_dict["n_epsilons"]]),
                  end="\\\\\\hline\n")
            for power_i, power in enumerate(self.train_param_dict["gaussian_powers"]):
                for dim_i, dim in enumerate(self.train_param_dict["n_gaussian_dims"]):
                    print(f"{power}&\t{dim}", end="")
                    for epsilon_i, epsilon in enumerate(self.train_param_dict["n_epsilons"]):
                        print(f"&\t{str(round(np.max(self.logli_average_over_samples[dim_i, power_i, epsilon_i]), 2))}",
                              end="")

                    print("\\\\", end="\\hline\n" if dim == 4 else "\n")
        else:
            print(f"B&\t"+"&\t".join([str(i) for i in self.train_param_dict["n_epsilons"]]),end = "\\\\\\hline\\hline\n")
            highest_logli_per_repeat = np.max(self.logli_average_over_samples,axis=1)
            print("Log-likelihood&\t"+"&\t".join([str(round(i,2)) for i in highest_logli_per_repeat]))
        pass

    def generate_halfmoons_forward_plot_matrix(self, plot_best_repeats = True):
        num_samples = 5000
        fig, ax = plt.subplots(1, 6, figsize=(14, 3))
        ax = ax.flatten()
        call = np.argmax if plot_best_repeats else np.argmin
        for epsilon_i, epsilon in enumerate(self.train_param_dict["n_epsilons"]):
            best_index = call(self.logli_average_over_samples[epsilon_i])
            model = self.get_model(((epsilon_i,epsilon),(best_index,best_index)))
            data = torch.randn((num_samples, model.dimension_of_flows)).to(self.device)
            inverse = model.inverse(data).detach().cpu().numpy()
            ax[epsilon_i].scatter(inverse[:,0], inverse[:,1], s=1, alpha = 0.2)
            ax[epsilon_i].set_title(f"B={epsilon}")
        target_data = build_px_samples("HALFMOONS", n_samples=num_samples).data
        ax[5].scatter(target_data[:,0],target_data[:,1],s=1,alpha=0.2,c="red")
        ax[5].set_title("Target")
        for current_ax in ax:
            current_ax.set_xlim((-1.5,2.5))
            current_ax.set_ylim((-1,1.5))
        plt.tight_layout()
        indicator = "best" if plot_best_repeats else "worst"
        plt.savefig(f"{self.folder}/forward_plot_matrix_{indicator}.png")
        plt.show()

    def generate_gaussian_forward_plot_matrix(self, plot_best_repeats = True):
        fig, ax = plt.subplots(3, 6, figsize=(12, 5))
        # For 2-D
        dim = 2
        dim_i = 0
        index_of_target_plot = 5
        plot_axis_limits_per_power = [(-3, 3), (-1, 5), (-7, 7)]
        num_samples = 5000
        call = np.argmax if plot_best_repeats else np.argmin

        # For each power

        for power_i, power in enumerate(self.train_param_dict["gaussian_powers"]):
            # For each epsilon
            for eps_i, epsilon in enumerate(self.train_param_dict["n_epsilons"]):
                best_repeat = call(self.logli_average_over_samples[dim_i, power_i, eps_i])
                name_of_model = f"GAUSS_{dim}dim_{power}pow_{epsilon}eps_{best_repeat}rep"
                checkpoint = torch.load(self.folder + "/" + name_of_model + ".p", map_location=self.device)
                model = marginalizingFlow(dim, epsilon, self.model_param_dict["n_layers"])
                model.load_state_dict(checkpoint["model"])
                data = torch.randn((num_samples, model.dimension_of_flows)).to(self.device)
                inverse = model.inverse(data).detach().cpu().numpy()
                ax[power_i, eps_i].scatter(inverse[:, 0], inverse[:, 1], s=1, alpha=0.2)
                ax[power_i, eps_i].set_xlim(plot_axis_limits_per_power[power_i])
                ax[power_i, eps_i].set_ylim(plot_axis_limits_per_power[power_i])
                ax[power_i, eps_i].set_title(f"ψ={power}, B={epsilon}")

                # Find the index of the best performing model
                # Make a plot
        for power_i, power in enumerate(self.train_param_dict["gaussian_powers"]):
            target_data = torch.randn((num_samples, 2)) ** power
            ax[power_i, index_of_target_plot].scatter(target_data[:, 0], target_data[:, 1], s=1, c="red", alpha=0.2)
            ax[power_i, index_of_target_plot].set_xlim(plot_axis_limits_per_power[power_i])
            ax[power_i, index_of_target_plot].set_ylim(plot_axis_limits_per_power[power_i])
            ax[power_i, index_of_target_plot].set_title(f"Target for ψ={power}")
        plt.tight_layout()
        indicator = "best" if plot_best_repeats else "worst"
        name = self.model_param_dict["dataname"]

        plt.savefig(f"{self.folder}/{name}_forward_plot_matrix_{indicator}.png")
        plt.show()
        # Make a target plot

    def generate_image_forward_plot_matrix(self, dataname):
        mnist = dataname.endswith("MNIST")
        n_epsilons = self.train_param_dict["n_epsilons"]
        n_samples_per_epsilon = 10
        best_repeats = torch.argmax(self.logli_average_over_samples, dim=0)
        target_index = 6
        fig, ax = plt.subplots(len(n_epsilons)+1, n_samples_per_epsilon, figsize= (15,10))
        for epsilon_i, epsilon in enumerate(n_epsilons):
            model = self.get_model(((epsilon_i,epsilon),(best_repeats[epsilon_i],best_repeats[epsilon_i])))
            data = torch.randn((n_samples_per_epsilon,model.dimension_of_flows)).to(self.device)
            inverse = model.inverse(data).detach().cpu().numpy()[:,:model.data_dimensions]
            if mnist:
                inverse_reshaped = inverse.reshape((n_samples_per_epsilon,28,28))
            else:
                inverse_reshaped = inverse.reshape((n_samples_per_epsilon,3, 32,32))
                inverse_reshaped = np.swapaxes(np.swapaxes(inverse_reshaped, 1,3),1,2)
            for sample_i in range(n_samples_per_epsilon):
                if mnist:
                    ax[epsilon_i,sample_i].imshow(inverse_reshaped[sample_i], cmap="Greys")
                else:
                    ax[epsilon_i,sample_i].imshow(inverse_reshaped[sample_i])
                ax[epsilon_i,sample_i].set_xticks([])
                ax[epsilon_i,sample_i].set_yticks([])
        actual_data = build_px_samples(dataname).data
        random_indices = np.random.choice(actual_data.shape[0],n_samples_per_epsilon, replace = False)
        random_data = actual_data[random_indices]
        for index, data_point in enumerate(random_data):
            if mnist:
                ax[target_index,index].imshow(data_point, cmap="Greys")
            else:
                ax[target_index, index].imshow(data_point)
            ax[target_index, index].set_xticks([])
            ax[target_index, index].set_yticks([])

        plt.subplots_adjust(wspace=0.001, hspace=0.001)

        plt.tight_layout()
        name = self.model_param_dict["dataname"]

        plt.savefig(f"{self.folder}/{name}_forward_plot_matrix.png")

        plt.show()



