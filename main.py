from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")
    t = Taster(device, "/home/s1003731/thesis/Thesis/Runs/FMNIST_2020-7-19_21:36:47")
    t.compute_logli(precomputed=False)
    t.print_best_model_table()
    t.plot_max_logli()
    t.generate()

    for i, v in enumerate(["KMNIST", "CIFAR10"]):
        layers = 4 if i < 3 else 6
        noise = 0.15 if i < 3 else None
        baker = Baker(device, n_layers=layers, n_epochs=10, batch_size=128, lr=5e-4, n_repeats=10)
        folder = baker.bake(v, [0, 1, 2, 4, 8, 16], noise, 0.6)
        taster = Taster(device, folder)
        taster.compute_logli(precomputed=False)
        taster.print_best_model_table()
        taster.plot_max_logli()
        taster.generate()

    # print("Finished")
    # print("You can either bake some new models, or evaluate some old models")
