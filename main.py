from Bakery.Baker import Baker
from Tastery.Taster import Taster
from builder import build_px_samples

if __name__ == "__main__":
    baker = Baker(batch_size=128, n_epochs = 1024, n_repeats=5)
    baker.bake_gaussian_models(eps= [0, 1, 2, 3, 4],
                               dims= [2, 3, 4],
                               pow= [1, 2, 3],
                               make_plots = False)

    # taster = Taster("/home/guus/PycharmProjects/Thesis/Runs/gaussian_2020-07-14 14:45:23.830946",True)
    # taster.compute_neglogli(precomputed=False)
    # taster.plot_avg_logli()
    # taster.plot_min_logli()
    # baker = Baker(n_layers=4, n_epochs=6)
    #
    # baker.bake("MNIST",[0,1,2,3,4], noise = 0.2, clipNorm = 0.4, make_plots=True)
    #
    # print("You can either bake some new models, or evaluate some old models")