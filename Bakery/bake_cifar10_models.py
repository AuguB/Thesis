from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.mnist_trainer import MNISTTrainer
import pickle
from time import *
import matplotlib.pyplot as plt
#
# n_epsilons = [0]
# n_layers = 4
# n_epochs = 1
# batch_size = 1024
# lr = 5e-4
# decay = 0.99
# n_pixels = 3*32*32



n_epsilons = [0, 1, 2, 4, 8, 16, 32, 64]
n_layers = 4
n_epochs = 5
batch_size = 16
lr = 5e-4
decay = 0.99
n_pixels = 3*32*32

start = time()
# Create a folder to store the test results
current_test_folder = make_top_folder(f"CIFAR10{n_layers}")
pickle.dump(n_epsilons,open(f"{current_test_folder}/n_epsilons.p","wb"))


# An object to hold the models, maps from epsilons to model
model_dict = {}
rep_losses = []
data = build_px_samples(100,0,"CIFAR10")
plt.imshow(data.data[0])
plt.show()
for eps_i, epsilon in enumerate(n_epsilons):
    flow = marginalizingFlow(n_pixels, epsilon, n_layers=n_layers, mnist=True)
    trainer = MNISTTrainer()
    losses = trainer.train(net=flow, dataset=data, n_epochs=n_epochs, batch_size=batch_size, lr=lr, decay = decay,dataname="CIFAR10")
    rep_losses.append(losses)
    plt.plot(losses)

    model_dict[epsilon] = flow.state_dict()

models_filename = "/".join([current_test_folder, "model_dict.p"])
pickle.dump(model_dict, open(models_filename, "wb"))
losses_filename = "/".join([current_test_folder, "loss_list.p"])
pickle.dump(rep_losses, open(losses_filename, "wb"))

stop = time()
duration = stop - start
print(duration, " seconds")




