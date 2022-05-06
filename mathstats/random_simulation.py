"""
import du module numpy
"""
import numpy as np

def transform_sampling(qantile, lambada, nobs):
    """
    La fonction permet de simuler des variables aleatoires avec
    la methode d'inverse de la fonction repartition
    """
    varx = np.zeros(nobs)
    for i in range(nobs):
        varx[i] = qantile(np.random.uniform(0, 1, 1), lambada)
    return varx


def centrales_limites(sets, size_n, nsim, mux, sigma):
    """
    La fonction retourne des variables aleatoires simulées avec
    la methode du central limite
    """
    varz = np.zeros(nsim)
    for i in range(nsim):
        varx = np.random.choice(sets, size=size_n, replace=False)
        varz[i] = np.sqrt(size_n)*(np.mean(varx) - mux)/sigma
    return varz
