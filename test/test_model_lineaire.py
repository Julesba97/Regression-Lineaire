"""
Import des modules
"""
import numpy as np
from mathstats.model_lineaire import LinearModel, Ridge, read_dataset




dataset = read_dataset('fuel2001.txt')
X = dataset[["Income", "Miles", "Tax", "MPC"]]
y = 1000 * dataset.FuelC / dataset.Pop

mod_1 = LinearModel(intercept=True)
mod_1.fit(X, y)
mod_1.summary(X, y)


def test_intercept_is_bool():
    """
    Tester le type de intercept
    """
    assert isinstance(mod_1.intercept, bool)


def test_rank_model():
    """
    Tester le rang du modele
    """
    assert mod_1.rank == mod_1.beta.shape[0]


def test_xtx_symetrie():
    """
    Verifier si XTX est reellement symetrique
    """
    assert np.array_equal(mod_1.transp(X), mod_1.transp(X).transpose())


mod_2 = Ridge(intercept=True, lambada=0)
mod_2.fit(X, y)
mod_2.summary(X, y)

mod_3 = Ridge(intercept=True, lambada=4000)
mod_3.fit(X, y)
mod_3.summary(X, y)

def test_ridge_equal_linearmodel_if_lambda_nul():
    """
    Tester si Rigde est equivalent à LinearModel lorque lambda=0
    """
    assert np.array_equal(mod_1.beta, mod_2.beta)

def test_ridge_differ_linearmodel_if_lambda_non_nul():
    """
    Tester si Rigde est different de LinearModel quand lambda est non nul
    """
    assert np.array_equal(mod_1.beta, mod_3.beta) is False
