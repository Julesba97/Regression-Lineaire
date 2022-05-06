""""
Ceci est le chargement des modules necessaires
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import pylab

#pylint: disable=too-many-locals
#pylint: disable=too-many-statements


def read_dataset(fichier):
    """
    La focntion reçoit un fichier text et retourne un dataset
    de type pandas.
    """
    with open(fichier, encoding="utf-8") as fic_op:
        read_line = fic_op.readlines()
        nobs = len(read_line)
        ncols = len(read_line[1].split())
        dic = {}
        for j in range(ncols):
            col = []
            for i in range(nobs):
                obs = read_line[i].split()
                col.append(obs[j])
            if j != ncols-1:
                dic[col[0]] = [float(x) for x in col[1:]]
            else:
                dic[col[0]] = col[1:]
    return pd.DataFrame(dic)


def cholesky(matrix):
    """
    Cette fonction prenant en entrée une matrice symetrique définie positive
    retourne une matrice triangulaire inférieur. C'est la methode de Cholesky
    """
    matrix = np.array(matrix)
    chol = np.zeros(matrix.shape)
    for j in range(matrix.shape[0]):
        for i in range(j, matrix.shape[0]):
            if i == j:
                chol[i, j] = np.sqrt(matrix[i, j] - np.sum(chol[i, :j]**2))
            else:
                chol[i, j] = (matrix[i, j] - np.sum(chol[i, :j]*chol[j, :j]))/chol[j, j]
    return chol


def solve_line_sys(chol, tary):
    """
    La fonction prend en entrée une matrice triangulaire inférieur et une matrice ligne.
    Le principe est de faire une resolution d'un système linéaire A*B=targy et le but
    est de trouver les valeurs de B.
    """
    chol = np.array(chol)
    tary = np.array(tary)
    tmatrix = chol.transpose()
    nline = chol.shape[0]
    soly = np.zeros(nline)
    solx = np.zeros(nline)
    for i in range(nline):
        sumj = 0
        for j in range(i):
            sumj += chol[i, j]*soly[j]
        soly[i] = (tary[i]-sumj)/chol[i, i]
    for i in sorted(range(nline), reverse=True):
        sumj = 0
        for j in range(i+1, nline):
            sumj += tmatrix[i, j]*solx[j]
        solx[i] = (soly[i]-sumj)/tmatrix[i, i]
    return solx.reshape((solx.shape[0], 1))


def matrix_dot(matrixa, matrixb):
    """
    La fonction prend en entrée deux matrice et de faire la multiplication
    """
    matrixa = np.array(matrixa)
    matrixb = np.array(matrixb)
    matrixc = np.zeros((matrixa.shape[0], matrixb.shape[1]))
    if matrixa.shape[1] == matrixb.shape[0]:
        for i in range(matrixa.shape[0]):
            for j in range(matrixb.shape[1]):
                for mrg in range(matrixa.shape[1]):
                    matrixc[i, j] = matrixc[i, j] + matrixa[i, mrg] * matrixb[mrg, j]
    return matrixc

def matrix_transp(matrix):
    """
    La fonction prend en entrée une matrice et de retourner sa transposée.
    """
    matrix = np.array(matrix)
    matrixa = np.zeros((matrix.shape[1], matrix.shape[0]))
    for i in range(matrixa.shape[0]):
        for j in range(matrixa.shape[1]):
            matrixa[i, j] = matrix[j, i]
    return matrixa



class OrdinaryLeastSquares:
    """ la classe permet d'estimer les coefficients du modele
    en utilisant la methode des moindres carres"""
    def __init__(self, intercept=True):
        self.intercept = intercept
        self.coeff = {}
        self.beta = np.array([])



    def transp(self, predictor):
        """ la focntion permet juste de multiplier la transpose de la matrice avec
         la matrice elle meme"""
        predictor1 = predictor.copy()
        if self.intercept is True:
            predictor1.insert(0, 'Const', list(np.ones(predictor.shape[0])))
            txx = matrix_dot(matrix_transp(np.array(predictor1)), np.array(predictor1))
        else :
            txx = matrix_dot(matrix_transp(np.array(predictor1)), np.array(predictor1))
        return txx


    def fit(self, predictor, target):
        """ la fonction permet d'estimer les coefficients"""
        predictor1 = predictor.copy()
        target = np.array(target).reshape((target.shape[0],1))
        if self.intercept is True:
            predictor1.insert(0, 'Const', list(np.ones(predictor.shape[0])))
            chol = cholesky(self.transp(predictor))
            txy = matrix_dot(matrix_transp(predictor1), target)
            self.beta = solve_line_sys(chol, txy)
        else:
            chol = cholesky(self.transp(predictor1))
            txy = matrix_dot(matrix_transp(predictor1), target)
            self.beta = solve_line_sys(chol, txy)
        for i in range(self.beta.shape[0]):
            self.coeff[predictor1.columns[i]] = self.beta[i, 0]


    def predict(self, predictor):
        """La fonction permet de donner la valeur du target en prenant
        comme entre les predicteurs """
        predictor1 = predictor.copy()
        if self.intercept is True:
            predictor1.insert(0, 'Const', list(np.ones(predictor.shape[0])))
            predic = matrix_dot(predictor1, self.beta)
        else:
            predic = matrix_dot(predictor1, self.beta)
        return predic


    def get_coeffs(self):
        """
        La fonction permet de  retourner les coefficients estimees
        avec leur coefficients associees
        """
        return self.coeff

class LinearModel(OrdinaryLeastSquares):
    """ C'est la classe de la regression Lineaire """
    def __init__(self, intercept):
        self.residuals = float()
        self.rsquare = float()
        self.rank = float()
        self.name = 'Linear Regression Model'
        super().__init__(intercept)


    def summary(self, predictor, target):
        """
        Cette fonction retourne un resumé des resultats de la regression lineaire
        """
        predictor1 = predictor.copy()
        target = np.array(target).reshape((target.shape[0], 1))
        targbar = np.mean(target)*np.ones((target.shape[0], 1))
        if self.intercept is True:
            matix = super().transp(predictor)
            self.residuals = target - super().predict(predictor)
            sct = sum((target-targbar)**2)[0]
            self.rank = np.linalg.matrix_rank(matix)
        else:
            self.residuals = target - super().predict(predictor)
            sct = sum(target**2)[0]
            matix = super().transp(predictor)
        inver = np.linalg.inv(matix)
        self.rank = np.linalg.matrix_rank(matix)
        scr = sum(self.residuals**2)[0]
        self.rsquare = (1 - scr/sct)
        nobs = np.array(predictor).shape[0]
        sigma = scr/(nobs-self.rank)
        std = []
        tstud = []
        pval = []
        borinf = []
        borsup = []
        for i in range(self.rank):
            std.append(np.sqrt(sigma*inver[i, i]))
            tstud.append(self.beta[i, 0]/std[i])
            pval.append(stats.t.sf(np.abs(tstud[i]), nobs - self.rank)*2)
            borinf.append(self.beta[i, 0] - stats.t.ppf(0.975, nobs - self.rank) * std[i])
            borsup.append(self.beta[i, 0] + stats.t.ppf(0.975, nobs - self.rank) * std[i])
        dic = {}
        dic["Estimate"] = list(self.beta.reshape(self.rank,))
        dic["Std err"] = std
        dic["t"] = tstud
        dic["P>|t|"] = pval
        dic["BorneInf"] = borinf
        dic["BorneSup"] = borsup
        if self.intercept is True:
            predictor1.insert(0, 'Const', list(np.ones(predictor.shape[0])))
            fsta = ((nobs-self.rank)/(self.rank - 1))*(self.rsquare/(1 - self.rsquare))
            pvalf = 1 - stats.f.cdf(fsta, self.rank-1, nobs-self.rank)
            raj = 1 - ((nobs-1)/(nobs-self.rank))*(1 - self.rsquare)
        else:
            raj = 1 - (nobs/(nobs-self.rank))*(1-self.rsquare)
        data = pd.DataFrame(dic, index=predictor1.columns)
        print(" "*37, self.name)
        print("*"*90)
        print("Residuals: ")
        dics= {}
        sre = lambda i: np.round(np.quantile(self.residuals,
                                             np.array([0, 0.25, 0.5, 0.75, 1]))[i], 2)
        dics["Min"] = sre(0)
        dics[" "*8] = " "
        dics["1Q"] = sre(1)
        dics[" "*7] = " "
        dics["Median"]  = sre(2)
        dics[" "*6] = " "
        dics["3Q"] = sre(3)
        dics[" "*5] = " "
        dics["Max"] = sre(4)
        print(pd.DataFrame([dics]).set_index("Min"))
        print("*"*90)
        print("Coefficients: ")
        print(data)
        print("*"*90)
        print("Residual Std err:", f"{round(np.sqrt(sigma), 4)}",
              " avec ", f'{nobs-self.rank}', " degré de liberté")
        print("R_square: ", f'{round(self.rsquare, 4)}',
              " "*10, 'Adj. R_square :', f"{round(raj, 4)}")
        if self.intercept is True:
            print("F-statistic :", f'{round(fsta,4)}', " avec", f"{self.rank-1}",
                  "et", f"{nobs-self.rank}", "DF", "p_valeur:", f'{pvalf}')
        print("*"*90)


    def determination_coefficient(self):
        """
        cette fonction permet de retourner le coefficient
        de determination du modele
        """
        return self.rsquare


    def graphe_residus(self):
        """
        Cette fonction permet d'afficher le graphe des résidus et à
        coté  le graphe qqpot qui permet d'observer l'adéquation des
        residus avec la normale.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.distplot(self.residuals, color='orange')
        plt.title('Distribution des résidus')
        plt.subplot(1, 2, 2)
        stats.probplot(self.residuals[:, 0], dist="norm", plot=pylab)
        pylab.show()
        plt.show()


class Ridge(LinearModel):
    """
    C'est la classe du modele Ridge.
    """
    def __init__(self, intercept, lambada):
        super().__init__(intercept)
        self.lambada = lambada
        self.name = "Ridge Model"


    def transp(self, predictor):
        """
        Cette fonction permet de mettre en jour la classe OLS en presence
        de la valeur lambda
        """
        return super().transp(predictor) + self.lambada * np.eye(super().transp(predictor).shape[0])
