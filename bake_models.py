from builder import *
from marginalizingflow import marginalizingFlow
from plotters import plot_mappings, plot_losses, plot_performance, show_forward
from trainer import *
import pickle
from time import *
import matplotlib.pyplot as plt
from torch.distributions import multivariate_normal, uniform
n_samples = 2048


n_samples = 2048
n_epsilons = [0, 1]
n_gaussian_dims = [2]
gaussian_powers = [1]
gaussian_max_shifts = [0]
n_layers = 8
n_epochs = 5
batch_size = 128
lr = 1e-4
decay = 0.997
n_repeats = 1
#
# n_samples = 2048
# n_epsilons = [0, 1, 2, 4, 8,16]
# n_gaussian_dims = [2, 4, 8,16]
# gaussian_powers = [1, 2, 3,4]
# gaussian_max_shifts = [0,0.25,0.5,1]
# n_layers = 8
# n_epochs = 1024
# batch_size = 128
# lr = 1e-4
# decay = 0.997
# n_repeats = 5

start = time()
# Create a folder to store the test results
current_test_folder = make_top_folder()

# An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
loss_interval = 32
loss_observations = math.ceil(n_samples / (batch_size * loss_interval)) * n_epochs
loss_dict = {}

# An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
model_dict = {}

# Files which make it easy to find the losses and the models
create_info_files(current_test_folder,
                  n_gaussian_dims,
                  n_epsilons,
                  n_repeats,
                  gaussian_max_shifts,
                  gaussian_powers)

total_number_of_runs = len(n_gaussian_dims) \
                       * len(n_epsilons) \
                       * n_repeats \
                       * len(gaussian_max_shifts) \
                       * len(gaussian_powers)

print(f"Going to train {total_number_of_runs} models")
i = 0

for dim_i, gaussian_dim in enumerate(n_gaussian_dims):
    for eps_i, epsilon in enumerate(n_epsilons):
        for shift_i, shift in enumerate(gaussian_max_shifts):
            for pow_i, pow in enumerate(gaussian_powers):
                rep_losses = np.zeros((n_repeats, loss_observations))
                for rep in range(n_repeats):
                    model_signature = f"gauss_{gaussian_dim}dim_{epsilon}eps_{shift}shift_{pow}pow_{rep}rep"
                    print(f"Now running {model_signature}")
                    print(f"{round(100 * i / total_number_of_runs, 2)}% of all models trained   ")
                    i += 1
                    y = get_gaussian_samples(n_samples, gaussian_dim, shift, pow)
                    flow = marginalizingFlow(gaussian_dim, epsilon, n_layers=6)
                    trainer = Trainer()

                    losses = trainer.train(net=flow, dataset=y, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                                           decay=decay, model_signature=model_signature, loss_interval=loss_interval)

                    rep_losses[rep, :len(losses)] = losses

                    model_dict[(gaussian_dim, epsilon, shift, pow, rep)] = flow.state_dict()

                loss_dict[(gaussian_dim, epsilon, shift, pow)] = rep_losses.copy()

# Plot the losses
fig, ax = plt.subplots(1, 1)
lookback = 16
losses = [np.mean(losses[i:i + lookback]) for i in range(len(losses) - lookback)]
ax.plot(losses)
plt.title(f"{n_epochs}epochs {batch_size}batch {lr}lr {decay}decay")
plt.show()

models_filename = "/".join([current_test_folder, "model_dict.p"])
pickle.dump(model_dict, open(models_filename, "wb"))
losses_filename = "/".join([current_test_folder, "loss_dict.p"])
pickle.dump(loss_dict, open(losses_filename, "wb"))

stop = time()
duration = stop - start
print(duration, " seconds")
