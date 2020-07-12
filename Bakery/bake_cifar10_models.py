from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

from builder import *
from Flows.marginalizingflow import marginalizingFlow
from Gym.mnist_trainer import MNISTTrainer
import pickle
from time import *
import matplotlib.pyplot as plt
#
n_epsilons = [0,1,2,4,8,16,32,64]
n_layers = 4
n_epochs = 10
batch_size = 64
lr = 1e-3
decay = 0.995
n_pixels = 3*32*32

start = time()
# Create a folder to store the test results
current_test_folder = make_top_folder(f"CIFAR10{n_layers}")
pickle.dump(n_epsilons,open(f"{current_test_folder}/n_epsilons.p","wb"))


# An object to hold the models, maps from epsilons to model
model_dict = {}
rep_losses = []
data = build_px_samples(100,0,"CIFAR10")
print(data.data.shape)
for eps_i, epsilon in enumerate(n_epsilons):
    flow = marginalizingFlow(n_pixels, epsilon, n_layers=n_layers, mnist=True)
    optim = Adam(flow.parameters(), lr = lr)
    scheduler = ExponentialLR(optim, decay)
    trainer = MNISTTrainer()
    losses= trainer.train(net=flow, dataset=data, optim = optim, scheduler=scheduler, n_epochs=n_epochs, batch_size=batch_size, dataname="CIFAR10")
    rep_losses.append(losses)
    plt.plot(losses)
    checkpoint = {
                  "optim":optim.state_dict(),
                  "model":flow.state_dict(),
                  "scheduler":scheduler.state_dict()
                  }
    torch.save(checkpoint,"/".join([current_test_folder, f"CIFAR_{n_layers}layers_{epsilon}eps_dict.p"]))

losses_filename = "/".join([current_test_folder, "loss_list.p"])
pickle.dump(rep_losses, open(losses_filename, "wb"))

stop = time()
duration = stop - start
print(duration, " seconds")




