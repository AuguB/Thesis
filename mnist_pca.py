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

plt.plot([sum(pca.explained_variance_ratio_[:i]) for i in range(784)])
plt.ylabel("Explained Variance")
plt.xlabel("Principal Components")
plt.grid(True)
plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/MNIST_varplot.png")
plt.show()




