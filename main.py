from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")
    lovelace_runs = "/home/s1003731/thesis/Thesis/Runs"
    folders = ["CIFAR10_2020-7-21_0:10:34", "MNIST_2020-7-19_21:36:47", "KMNIST_2020-7-20_15:53:32",
               "MNIST_2020-7-19_16:2:16"]

    for f in folders:
        t = Taster(device, f"{lovelace_runs}/{f}")
        t.compute_logli(precomputed=True)
        t.plot_avg_logli()
        t.generate()

    # print("Finished")
    # print("You can either bake some new models, or evaluate some old models")
