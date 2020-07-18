from Bakery.Baker import Baker
from Bakery.Taster import Taster
import torch
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Going to train models on {device}")
    # taster = Taster(device, "/home/guus/PycharmProjects/Thesis/Runs/MNIST_2020-7-15_21:10:36")
    #
    # taster.compute_logli(precomputed=True)
    # taster.plot_max_logli()
    # taster.generate()

    #
    lovelace_runs_folder = "/home/s1003731/thesis/Thesis/Runs"
    print("MNIST")
    print()

    taster = Taster(device, f"{lovelace_runs_folder}/MNIST_2020-7-15_21:10:36")
    # taster.compute_logli(precomputed=False)
    # taster.print_best_model_table()
    # taster.plot_max_logli()
    taster.generate()

    print("FMNIST")
    print()

    taster = Taster(device, f"{lovelace_runs_folder}/FMNIST_2020-7-15_22:13:2")
    # taster.compute_logli(precomputed=False)
    # taster.print_best_model_table()
    # taster.plot_max_logli()
    taster.generate()

    print("KMNIST")

    print()

    taster = Taster(device, f"{lovelace_runs_folder}/KMNIST_2020-7-15_23:16:37")
    # taster.compute_logli(precomputed=False)
    # taster.print_best_model_table()
    # taster.plot_max_logli()
    taster.generate()


    # MNIST_baker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    # MNIST_baker.bake("MNIST",[0, 1, 2, 4, 8, 16],0.15,0.6)
    # FMNIST_baker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    # FMNIST_baker.bake("FMNIST", [0, 1, 2, 4, 8, 16], 0.15, 0.6)
    # KMNIST_baker = Baker(device,n_layers=4, n_epochs=10, batch_size=32, lr=5e-4)
    # KMNIST_baker.bake("KMNIST", [0, 1, 2, 4, 8, 16], 0.15, 0.6)
    # Gauss_baker = Baker(device,n_samples = 2048,n_layers=4, n_epochs=1024, batch_size=256, n_repeats=10)
    # Gauss_baker.bake_gaussian_models([0, 1, 2, 3, 4], [2,3,4],[1,2,3])
    # Halfmoon_baker = Baker(device,n_samples = 2048,n_layers=4, n_epochs=1024, batch_size=64, n_repeats=10)
    # Halfmoon_baker.bake("HALFMOONS", [0, 1, 2, 3, 4])
    # Cifar_baker = Baker(device,n_layers=6, n_epochs=16, batch_size=32, lr=5e-4)
    # Cifar_baker.bake("CIFAR10", [0, 1, 2, 4, 8, 16], clip_norm=0.6)

    # print("Finished")


    # print("You can either bake some new models, or evaluate some old models")
