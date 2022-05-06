"""
Chargement du module opera_matrix
"""
import numpy as np
import pandas as pd
from mathstats.model_lineaire import read_dataset, cholesky, matrix_transp, matrix_dot, solve_line_sys


matrix_1 = np.array([[8, 3.22, 0.8, 0.00, 4.10],
                     [3.22, 7.76, 2.33, 1.91, -1.03],
                     [0.8, 2.33, 5.25, 1, 3.02],
                       [0.00, 1.91, 1,7.5, 1.03],
                       [4.10, -1.03, 3.02, 1.03, 6.44]])
matrix_2 = np.array([[5.2, 3, 0.5, 1, 2],
                     [3, 6.3, -2, 4, 0],
                     [0.5, -2, 8, -3.1, 3],
                     [1, 4, -3.1, 7.6, 2.6],
                     [2, 0, 3, 2.6, 15]])
b_y = np.array([9.45, -12.2, 7.78, -8.10, 10])

def test_cholesky():
    """
    C'est pour tester la fonction cholesky
    """
    assert np.array_equal(np.linalg.cholesky(matrix_1), cholesky(matrix_1))


def test_matrix_dot():
    """
    C'est pour tester la fonction matrix_dot
    """
    assert np.array_equal(np.round(np.dot(matrix_1, matrix_2), 5),
                          np.round(matrix_dot(matrix_1, matrix_2), 5))

def test_matrix_transp():
    """
    C'est pour tester la fonction matrix_transp
    """
    assert np.array_equal(matrix_2.transpose(), matrix_transp(matrix_2))

def test_solve_line_sys():
    """
    C'est pour tester la fonction solve_line_sys
    """
    chol = cholesky(matrix_2)
    assert np.array_equal(np.round(solve_line_sys(chol, b_y), 5),
                          np.round(np.linalg.solve(matrix_2, b_y).reshape((5, 1)), 5))

def test_read_dataset():
    """
    C'est pour tester la fonction read_dataset
    """
    dataset_pandas = pd.read_csv("../mathstats/fuel2001.txt", sep=" ")
    dataset_read = read_dataset("fuel2001.txt")
    assert np.array_equal(np.array(dataset_pandas.drop("State", axis=1)),
                          np.array(dataset_read.drop("State", axis=1)))
