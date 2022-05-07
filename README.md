---
output:
  pdf_document: default
  html_document: default
---
# Implémentaion d'un package de modele linéaire
## Description
L'objectif de ce projet est de créer un package avec le langage python qui permet de
résoudre un problème de regression linéaire des moindres carrés ordinaires. Notre package, nommé `mathstats` fournit des classes
et des fonctions pour estimer les coefficients du modèle. Ensuite il réalise des tests statistiques
et affiche le résumé statistique du modèle. En fin il permet de faire des prédictions et 
visualiser le graphe des résidus pour voir l'ajustement du modèle sur les données. Le package est construit 
sous forme d'un style R et nos résultats sont testés par rapport au logiciel Rstudio 
pour s'assurer qu'ils sont corrects.
Ces fonctions ainsi que ces classes seront étudiées plus amplement dans la suite.

## Modules

### 1. [model_lineaire]()

Pour surmonter l'une des contraintes imposées qui était de minimiser au maximum l'utilisation des
librairies existants comme numpy et pandas. Alors nous avons coder bout à bout la factorisation de 
cholesky et une résolution d'un sytème linéaire associée. Ensuite il s'ensuit l'implémentation d'une méthode
de produit matriciel et une autre fonction pour transposer une matrice. En fin au lieu d'utiliser 
la fonction `read_csv` de pandas nous avons créé notre propre fonction pour lire les données.

> **cholesky(A)**
> 
> `paramétres:` A: matrice symétrique définie positive
> 
> `retour:` L: une matrice triangulaire inférieure
> 
> C'est la factorisation de Cholesky qui est appliquée ici. La methode consiste à dire 
> pour toute matrice symétrique définie positive on peut déterminer une matrice triangulaire inférieure L telle que:
>  A=LxLT avec LT transposé de L.

> **solve_line_sys(L,y)**
> 
> La fonction permet de résoudre une équation matricielle linéaire sous cette
> forme ax=y. avec a : symétrique définie positive.
>
>`paramètres:` 
> L: matrice triangulaire inférieure(retour de Cholesky) et y: une matrice colonne 
>       
> `retour:` solution du système ax=y, x a la même forme que y                 

> **matrix_dot(A,B)**
> 
> Cette fonction permet de faire le produit de deux matrices
> 
> `paramètres:` A(n,p) et B(p,m) 
> 
> `retour:` une matrice de taille (n,m)

> **matrix_transp(A)**
> 
> La fontion  permet de transposer une matrice
> 
> `paramètres:` A matrice de taille (n,p)
> 
> `retour:` AT matrice de taille (p,n)

> **read_dataset(fichier)**
> 
> La fonction permet de lire un fichier type texte
> 
> `paramètres:` fichier format textuelle
> 
> `retour:` DataFrame


Ce module contient aussi la classe **OrdinaryLeastSquares** qui utilise la méthode 
des moindres carrés pour estimer les paramètres du modèle.

> **OrdinaryLeastSquares(intercept=True)**
> 
> `paramètres:` intercept = True par défaut , type: booléen 
> 
> `attributs:` intercept, coeff: contient les coefficients estimés associés aux variables, beta: l'estimateur des moindres carrés
> 
> `méthodes:` 
> 
> **transp(X)**: 
> 
> `paramètres:` prend des données X, type: DataFrame
> 
> `retour:` retourne XTX type: ndarray (où XT transposé de X) 
> 
> Une fonction dont le motif est de permettre de mise en jour la classe lorsque nous implémenterons la classe Ridge.
> 
>**fit(X,y)** : 
> 
> `paramètres:`  prend des données X,y, type: DataFrame
> 
>  `retour:` pas de retour
> 
> La fonction prend des données X,y en entrée pour ajuster le modèle et calcul l'estimateur du modèle.
> 
> **predict(X)**:
> 
> `paramètres:` prend des données X, type: DataFrame
> 
> `retour:`  retourne les valeurs prédictes y, type: ndarray 
> 
> La fonction permet de faire la prédiction  du modèle de regression linéaire
> 
>**get_coeffs( )**:
> 
> retourne les coefficients estimés du modèle associés au nom des variables.
> 

Dans ce module on y trouve ensuite la classe **LinearModel** qui hérite de la classe
**OdinaryLeastSquares**. L'idée que nous avons est de considérer cette dernière comme un outil d'estimation
des coefficients du modèle linéaire. Alors **LinearModel** aura les mêmes attributs et
les mêmes méthodes que **OrdianryLeastSquares**. Dans la suite nous allons décrire les fonctionnalitées qui lui sont propre.

> **LinearModel(OrdinaryLeastSquares)**
> 
> `paramètres:` intercept , type: booléen
> 
> `attributs:` residuals: residus du modèle, rsquare: coefficient de determination, rank: rang des prédicteurs, name: nom du modèle 
> 
> `méthodes:`
> 
> **summary(X,y)**
> 
> `paramètres:` prend en entrée les données X et y type: DataFrame
> 
> `retour:` retourne une liste de statistiques récapitulatives du modèle
> 
> **determination_coefficient( )**
> 
> renvoie le coefficient de determination du modèle
> 
> **graphe_residus( )**
> 
> retourne la distribution des residus et un graphe qqplot associé. Ceci permet d'avoir une idée sur l'ajustement
> du modèle aux données.
>

Ce module contient enfin la classe **Ridge** qui a pour but de régulariser l'estimateur des moindres carrés.
Cette classe hérite de la classe **LinearModel**.

> **Ridge(LinearModel, intercept=True, lambda )**
> 
> `paramètres:` intercept: booléen, lambda: float
> 
> `attributs:`  intercept, lambda
> 
> `méthodes:` 
> 
> **transp(X, lambda)**
> 
> `paramètres:` des données X: DataFrame, lambda: float
> 
> `retour:` retourne un tableau, type: ndarray
> 
> Cette fonction  vise à mettre en jour la classe **OrdinaryLeastSquares** pour faire l'estimation des coefficients du modèle.
> 


### 2. [random_simulation]()

Dans ce module il était question d'implémenter des fonctions qui 
simulent des variables aleatoires.

> **transform_sampling(quantile, n, lambda)**
> 
> La fontionn simule des variables aléatoires en utilisant l'inverse généralisée de la fonction de répartition F.
> 
> `paramètres:` quantile: inverse généralisé de F, n: nombre de variables aléatoires et lambda parametre de la loi exponentielle
> 
>`retour:` variables aleatoires dont sa loi est caractérisé par F

>**centrales_limites(sets, size_n, nsim, mux, sigma)**
> 
> Cette fonction simule des variables aléatoires suivant une loi normale centré réduite.
> 
> `paramètres:` sets: ensemble de variables aleatoirs, size_n: nombre de variable choisie dans sets, nsim : nombre de variables retournées, mux : moyenne, sigma : ecart type 
> 
> `retour:` nsim variables aleatoires normales centrées réduites
