from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")
    # lovelace_runs = "/home/s1003731/thesis/Thesis/Runs"
    speedy_runs = "/home/guus/Thesis/Runs"
    folders = ["CIFAR10_2020-7-21_0:10:34", "MNIST_2020-7-19_16:2:16", "KMNIST_2020-7-20_15:53:32",
               "FMNIST_2020-7-19_21:36:47"]

    for i,f in enumerate(folders):
        t = Taster(device, f"{speedy_runs}/{f}")
        t.compute_logli(precomputed=True)
        print("test1.0")
        # t.plot_avg_logli()
        t.plot_max_logli()
        t.print_best_model_table()
        print("test1.1")
        t.generate()
        print("test1.2")

    # print("Finished")
    # print("You can either bake some new models, or evaluate some old models")
