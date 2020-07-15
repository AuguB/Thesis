from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")

    MNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    MNISTbaker.bake("MNIST",[0, 1, 2, 4, 8, 16],0.2,0.6)
    FMNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    FMNISTbaker.bake("FMNIST", [0, 1, 2, 4, 8, 16], 0.2, 0.6)
    KMNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    KMNISTbaker.bake("KMNIST", [0, 1, 2, 4, 8, 16], 0.2, 0.6)
    Gaussbaker = Baker(device,n_layers=4, n_epochs=2048, batch_size=256, n_repeats=10)
    Gaussbaker.bake([0, 1, 2, 3, 4], [2,3,4],[1,2,3])
    Halfmoonbaker = Baker(device,n_layers=4, n_epochs=2048, batch_size=16, n_repeats=10)
    Halfmoonbaker.bake("HALFMOONS", [0, 1, 2, 3, 4])
    Cifarbaker = Baker(device,n_layers=6, n_epochs=16, batch_size=32, lr=5e-4)
    Cifarbaker.bake("CIFAR10", [0, 1, 2, 4, 8, 16],clipNorm=0.6)

    print("Finished")

    # print("You can either bake some new models, or evaluate some old models")
