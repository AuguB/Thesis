from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from builder import get_MNIST

data = get_MNIST(train=True).data.reshape((-1,784))

data_sd = StandardScaler().fit_transform(data)

pca = PCA()

pca_fit= pca.fit_transform(data_sd)

cums = [sum(pca.explained_variance_ratio_[:i]) for i in range(784)]
for i in range(784):
    if cums[i] > 0.95:
        print(i,"95")
        break
for i in range(784):
    if cums[i] > 0.999:
        print(i, "99")
        break


fig,ax = plt.subplots(2,1,figsize=(7,10), sharex=True)
ax[0].plot(pca.explained_variance_ratio_, label = "Explained variance per PC",color = "red")
ax[0].set_ylabel("Explained Variance")
ax[0].set_xlabel("Principal Components")
ax[0].grid(True)
ax[0].legend()

ax[1].plot([sum(pca.explained_variance_ratio_[:i]) for i in range(784)], label="Cumulative explained variance", color = "blue")
ax[1].set_ylabel("Explained Variance")
ax[1].grid(True)
ax[1].legend()

plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/MNIST_varplot.png")
plt.legend()
plt.show()




