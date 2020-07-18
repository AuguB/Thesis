from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from builder import get_MNIST, get_FMNIST, get_KMNIST

data = get_MNIST(train=True).data.reshape((-1,784))

data_sd = StandardScaler().fit_transform(data)

pca = PCA()

pca_fit= pca.fit_transform(data_sd)


fig,ax = plt.subplots(2,1,figsize=(4,5), sharex=True)

data = get_MNIST(train=True).data.reshape((-1,784))
data_sd = StandardScaler().fit_transform(data)
pca = PCA()
pca_fit= pca.fit_transform(data_sd)

ax[0].plot(pca.explained_variance_ratio_,label="MNIST")
ax[0].set_ylabel("Explained Variance")
ax[0].set_xlabel("Principal Components")
ax[0].grid(True)

ax[1].plot([sum(pca.explained_variance_ratio_[:i]) for i in range(784)], label="MNIST")
ax[1].set_ylabel("Explained Variance")
ax[1].grid(True)

data = get_FMNIST(train=True).data.reshape((-1,784))
data_sd = StandardScaler().fit_transform(data)
pca = PCA()
pca_fit= pca.fit_transform(data_sd)

ax[0].plot(pca.explained_variance_ratio_,label="FMNIST")
ax[1].plot([sum(pca.explained_variance_ratio_[:i]) for i in range(784)], label="FMNIST")

data = get_KMNIST(train=True).data.reshape((-1,784))
data_sd = StandardScaler().fit_transform(data)
pca = PCA()
pca_fit= pca.fit_transform(data_sd)

ax[0].plot(pca.explained_variance_ratio_,label="KMNIST")
ax[1].plot([sum(pca.explained_variance_ratio_[:i]) for i in range(784)], label="KMNIST")
ax[1].legend()
ax[0].legend()


plt.savefig("/home/guus/PycharmProjects/Thesis/Plots/MNIST_varplot.png")
plt.legend()
plt.show()




