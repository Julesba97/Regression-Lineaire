
"""
Chargement des packages necéssaires
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from model_lineaire import LinearModel, Ridge, read_dataset
from random_simulation import centrales_limites, transform_sampling


dataset = read_dataset('fuel2001.txt')
X = dataset[["Income", "Miles", "Tax", "MPC"]]
X["Dlic"] = 1000 * dataset.Drivers / dataset.Pop
y = 1000 * dataset.FuelC / dataset.Pop
mod = LinearModel(intercept=False)
mod.fit(X, y)
mod2 = Ridge(intercept=False, lambada=4000)
mod2.fit(X, y)


tcl_z2 = centrales_limites(np.random.uniform(0, 1, 100000),
                           size_n=100, nsim=2000, mux=0.5, sigma=np.sqrt(1/12))
tcl_z1 = centrales_limites(np.random.poisson(5, 100000),
                           size_n=100, nsim=2000, mux =5, sigma=np.sqrt(5))
abscisse = np.linspace(-10, 10)


qantiexp = lambda u, lambada: -np.log(1-u)/lambada
sample1 = transform_sampling(qantiexp, 1, 10000)
abscisse1 = np.linspace(0, 10)


if __name__=="__main__":
    print(mod.summary(X, y))
    print(mod.graphe_residus())

    print(mod2.summary(X, y))
    print(mod2.graphe_residus())

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.hist(tcl_z2, bins=50, color="yellow", density=True)
    plt.plot(abscisse, stats.norm.pdf(abscisse, loc=0, scale=1), color="black", lw=2)
    plt.title("TCL uniforme(0,1)")
    plt.xlabel("variables")
    plt.ylabel("counts")
    plt.subplot(1, 2, 2)
    plt.hist(tcl_z1, bins=50, color="yellow", density=True)
    plt.plot(abscisse, stats.norm.pdf(abscisse, loc=0, scale=1), color="black", lw=2)
    plt.title("TCL poissons(5)")
    plt.xlabel("variables")
    plt.ylabel("counts")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(sample1, bins=100, color="yellow", density=True)
    plt.plot(abscisse1, stats.expon.pdf(abscisse1, loc=0, scale=1), color="black", lw=2)
    plt.title("Distribution de la loi exponentielle")
    plt.xlabel("variables")
    plt.ylabel("counts")
    plt.show()