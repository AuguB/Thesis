from Bakery.Baker import Baker
from multiprocessing import Process, Lock

from Tastery.Taster import Taster

if __name__ == "__main__":
    KMNISTbaker = Baker(n_layers=2, n_epochs=1, batch_size=1024)
    KMNISTbaker.bake("KMNIST",[0],True)

    # taster = Taster("/home/guus/PycharmProjects/Thesis/Runs/FMNIST_2020-7-14_21:8:0", False)
    # taster.generate()

    # MNISTbaker = Baker(n_layers=4, n_epochs=4, batch_size=32)
    # FMNISTbaker = Baker(n_layers=4, n_epochs=4, batch_size=32)
    # KMNISTbaker = Baker(n_layers=4, n_epochs=4, batch_size=32)
    # GaussBaker = Baker(n_layers = 6, n_epochs=2048, batch_size = 128)
    # HalfmoonBaker = Baker(n_layers = 4, n_epochs=2048, batch_size = 64)
    #
    # processes = [
    #     Process(target=MNISTbaker.bake, kwargs={"n_layers":4,"n_epochs":6,})
    # ]


    # print("You can either bake some new models, or evaluate some old models")
