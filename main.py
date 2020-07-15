from Bakery.Baker import Baker
from Bakery.Taster import Taster
from multiprocessing import Process
import torch
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")

    MNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    FMNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    KMNISTbaker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    Gaussbaker = Baker(device,n_layers=4, n_epochs=2048, batch_size=256, n_repeats=10)
    Halfmoonbaker = Baker(device,n_layers=4, n_epochs=2048, batch_size=16, n_repeats=10)
    Cifarbaker = Baker(device,n_layers=6, n_epochs=16, batch_size=32, lr=5e-4)

    processes = [
        Process(target=MNISTbaker.bake,
                args=("MNIST", [0, 1, 2, 4, 8, 16]),
                kwargs={"noise": 0.2, "clipNorm": 0.6}),
        Process(target=FMNISTbaker.bake,
                args=("FMNIST", [0, 1, 2, 4, 8, 16]),
                kwargs={"noise": 0.2, "clipNorm": 0.6}),
        Process(target=KMNISTbaker.bake,
                args=("KMNIST", [0, 1, 2, 4, 8, 16]),
                kwargs={"noise": 0.2, "clipNorm": 0.6}),
        Process(target=Gaussbaker.bake_gaussian_models,
                args=([0, 1, 2, 3, 4], [2,3,4],[1,2,3])),
        Process(target=Halfmoonbaker.bake,
                args=("HALFMOONS", [0, 1, 2, 3, 4])),
        Process(target=Cifarbaker.bake,
                args=("CIFAR10", [0, 1, 2, 4, 8, 16]),
                kwargs={"clipNorm": 0.6})
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Finished")

    # print("You can either bake some new models, or evaluate some old models")
