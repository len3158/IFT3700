# Auteur : Lenny SIEMENI
# Implementation de l'algorithme K-moyennes

from sklearn import datasets
import numpy as np

# On charge le jeu de donnees MNIST
mnist = datasets.load_digits()

print(mnist.data)