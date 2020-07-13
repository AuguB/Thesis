from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.trainer_bak import *
import pickle
from time import *
import math

n_samples = 2048
n_epsilons = [0, 1, 2, 3, 4]
n_layers = 4
n_epochs = 1024
batch_size = 64
lr = 1e-2
decay = 0.997
n_repeats = 5

start = time()
# Create a folder to store the test results
current_test_folder = make_top_folder(f"halfmoons{n_layers}")

# An object to hold all the losses, maps from tuple (distribution, epsilons) to ndarray(n_repeats, loss_observations)
loss_interval = 32
loss_observations = math.ceil(n_samples / (batch_size * loss_interval)) * n_epochs
loss_dict = {}

# An object to hold the models, maps from tuple (distribution, epsilons, repeat) to model
model_dict = {}
total_number_of_runs = len(n_epsilons) * n_repeats
print(f"Going to train {total_number_of_runs} models")
i = 0

for eps_i, epsilon in enumerate(n_epsilons):
    rep_losses = np.zeros((n_repeats, loss_observations))
    for rep in range(n_repeats):
        model_signature = f"halfmoon_{epsilon}eps_{rep}rep"
        print(f"Now running {model_signature}")
        print(f"{round(100 * i / total_number_of_runs, 2)}% of all models trained   ")
        i += 1

        y = build_px_samples(n_samples, 0, "half_moons")
        flow = marginalizingFlow(2, epsilon, n_layers=6)
        trainer = Trainer_bak()

        losses = trainer.train(net=flow, dataset=y, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                               decay=decay, model_signature=model_signature, loss_interval=loss_interval)

        rep_losses[rep, :len(losses)] = losses

        model_dict[(epsilon, rep)] = flow.state_dict()

    loss_dict[epsilon] = rep_losses.copy()

# Plot the losses
# fig, ax = plt.subplots(1, 1)
# lookback = 16
# losses = [np.mean(losses[i:i + lookback]) for i in range(len(losses) - lookback)]
# ax.plot(losses)
# plt.title(f"{n_epochs}epochs {batch_size}batch {lr}lr {decay}decay")
# plt.show()

models_filename = "/".join([current_test_folder, "model_dict.p"])
pickle.dump(model_dict, open(models_filename, "wb"))
losses_filename = "/".join([current_test_folder, "loss_dict.p"])
pickle.dump(loss_dict, open(losses_filename, "wb"))

stop = time()
duration = stop - start
print(duration, " seconds")
