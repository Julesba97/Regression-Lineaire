"""
Chargement des packages necessaires
"""
import numpy as np
from mathstats.random_simulation import transform_sampling, centrales_limites


qantiexp = lambda u, lambada: -np.log(1-u)/lambada

def test_transform_sampling():
    """
    C'est pour tester la fonction transform_sampling
    """
    np.random.seed(500)
    random_numpy = np.random.exponential(1, 500)
    np.random.seed(500)
    random_sampling = transform_sampling(qantiexp, 1, 500)
    assert np.array_equal(random_sampling, random_numpy)

def test_centrales_limites():
    """
    C'est pour tester la fonction centrales_limites
    """
    obser_norm = np.random.normal(0, 1, 2000)
    obser_centr = centrales_limites(np.random.uniform(0, 1, 100000),
                                    size_n=100,
                                    nsim=2000, mux=0.5, sigma=np.sqrt(1/12))
    assert len(obser_norm) == len(obser_centr)
