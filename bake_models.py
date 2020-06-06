from builder import *
from marginalizingflow import marginalizingFlow
from plotters import plot_mappings, plot_losses, plot_performance, show_forward
from trainer import *
import pickle
from time import *
import matplotlib.pyplot as plt

n_samples = 2048
n_epsilons = [0, 1, 2, 4, 8, 16]
n_gaussian_dims = [2, 3, 4, 8, 16]
gaussian_powers = [1,2,3,4]
gaussian_max_shifts = [0, 0.5, 1, 1.5]
distributions = ["square_gaussian"]
n_layers = 6
n_epochs = 512
batch_size = 128
lr = 5e-3
decay = 0.997
n_repeats = 5

start = time()
# Create a folder to store the test results
current_test_folder = make_top_folder()

# An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
loss_interval = 32
loss_observations = math.ceil(n_samples / (batch_size * loss_interval)) * n_epochs
loss_dict = {}

# An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
model_dict = {}

# Files containing tuples which make it easy to find the losses and the models
create_info_files(current_test_folder, distributions, n_epsilons, n_repeats)

total_number_of_runs = len(distributions) * len(n_epsilons) * n_repeats
print(f"Going to train {total_number_of_runs} models")

i = 0

for nth_distribution, distribution in enumerate(distributions):

    for j, epsilon in enumerate(n_epsilons):

        epsilon_losses = np.zeros((n_repeats, loss_observations))

        for r in range(n_repeats):
            model_signature = f"{distribution}_{epsilon}pad_{r}try"

            print(f"Now running {model_signature}")

            print(f"{round(100 * i / total_number_of_runs, 2)}% of all models trained   ")

            i += 1

            data = build_px_samples(n_samples=n_samples, distribution=distribution)

            n_datapoints, data_dim = data.shape

            flow: marginalizingFlow = marginalizingFlow(data_dim, epsilon, n_layers)

            trainer = Trainer()

            losses = trainer.train(net=flow, dataset=data, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                                   decay=decay, model_signature=model_signature, loss_interval=loss_interval)

            epsilon_losses[r, :len(losses)] = losses

            model_dict[(distribution, epsilon, r)] = flow.state_dict()

        loss_dict[(distribution, epsilon)] = epsilon_losses.copy()

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
