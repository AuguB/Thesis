from Bakery.Baker import Baker
from builder import build_px_samples

if __name__ == "__main__":
    # baker = Baker(batch_size=256)
    # baker.bake_gaussian_models([0,1,2,3,4])

    baker = Baker(n_layers=4, n_epochs=3)
    baker.bake("MNIST",[0,1,2,3,4], noise = 0.2, clipNorm = 0.4, make_plots=True)

    print("You can either bake some new models, or evaluate some old models")