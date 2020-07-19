from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")

    for i in ["MNIST", "FMNIST", "KNNIST", "CIFAR10"]:
        baker = Baker(device, n_layers=4, n_epochs=10, batch_size=256, lr=5e-4, n_repeats=10)
        folder = baker.bake(i, [0, 1, 2, 4, 8, 16], 0.15, 0.6)
        taster = Taster(device, folder)
        taster.compute_logli(precomputed=False)
        taster.print_best_model_table()
        taster.plot_max_logli()
        taster.generate()

    # print("Finished")
    # print("You can either bake some new models, or evaluate some old models")
