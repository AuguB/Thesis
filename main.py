from Bakery.Baker import Baker
from Bakery.Taster import Taster
from multiprocessing import Process

if __name__ == "__main__":

    MNISTbaker = Baker(n_layers=4, n_epochs=6, batch_size=32, lr=5e-4)
    FMNISTbaker = Baker(n_layers=4, n_epochs=6, batch_size=32, lr=5e-4)
    KMNISTbaker = Baker(n_layers=4, n_epochs=6, batch_size=32, lr=5e-4)
    Gaussbaker = Baker(n_layers=4, n_epochs=2048, batch_size=256)
    Halfmoonbaker = Baker(n_layers=4, n_epochs=2048, batch_size=16)
    Cifarbaker = Baker(n_layers=6, n_epochs=6, batch_size=32, lr=5e-4)

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
