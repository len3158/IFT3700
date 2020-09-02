# Auteur : Lenny SIEMENI
# Implementation de la l'algorithme des K-Plus proches voisins en utilisant la distance de
# minkowski
# RAPPEL : MNIST dans Scikitlearn a :
	# 10 classes
	# Chaque image est de taille 8*8
	# 1797 echantillons d'images au total
	# donc 1797*64
	# ~ 180 echantillons par classe
	
# Pour avoir le jeu de donnees MNIST au complet
# Telecharger a partir de open_ml
# Ne pas oublier d'ajuster les bornes de partitionnement

from sklearn import datasets
import numpy as np

# Nos donnees etants representees sous forme matricielle
# On effectue sur le sous-jeu de donnee le calcul de la distance de minkowski
def minkowski_mat(x1, X2, p=2.0):
	dist = (np.abs(x1 - X2)**p).sum(axis=1)**(1.0/p)
	return dist

# Le coeur de l'algorithme
# Apres avoir determine la distance des voisins les plus proches
# On retourne la distance minimale entre tous
# Un changement sur la valeur de p affecte la performance mais aussi la precision
# determiner le meilleur p a choisir plus tard
# On cherche le point de donnée avec la distance la plus petite, puis on retourne sa classe
def knn(x, data, p=18):
	feats = data[:,:-1]
	targets = data[:,-1]
	dist = minkowski_mat(x, feats, p)
	return targets[np.argmin(dist)]
	
# On charge le jeu de donnees MNIST
mnist = datasets.load_digits()


# Initialisation de la matrice de predictions
predictions = np.zeros(mnist.data.shape[0])
for i in range(mnist.data.shape[0]):
	predictions[i] = knn(mnist.data[i,:-1],mnist.data)

indexes = np.arange(mnist.data.shape[0])

## Separation de l'ensemble d'apprentissage et de l'ensemble d'entrainement,
## 70% (apprentissage) vs 20% (entrainement)
train_set = mnist.data[indexes[:1257]]
test_set = mnist.data[indexes[1257:1617]]

## Reste 10% pour l'ensemble de validation
validation_set = mnist.data[indexes[1617:]]

# Predictions sur l'ensemble d'apprentissage
train_predictions = np.zeros(train_set.shape[0])
for i in range(train_set.shape[0]):
	train_predictions[i] = knn(train_set[i,:-1],train_set)

# Predictions sur l'ensemble d'entrainement
test_predictions = np.zeros(test_set.shape[0])
for i in range(test_set.shape[0]):
	test_predictions[i] = knn(test_set[i,:-1],train_set)
	
# Predictions sur l'ensemble de validation
validation_predictions = np.zeros(validation_set.shape[0])
for i in range(validation_set.shape[0]):
	validation_predictions[i] = knn(validation_set[i,:-1],train_set)

# Puisque nous sommes dans un apprentissage supervise (euuh a verifier ? ^^')
# On calcule le taux d'erreur entre la prediction et ce qui est attendu
cible = mnist.data[:,-1]
print("Taux d'erreur sur le jeu d'apprentissage :",(1.0-(train_predictions==train_set[:,-1]).mean())*100.0)
print("Taux d'erreur sur le jeu d'entrainement : ", (1.0-(test_predictions==test_set[:,-1]).mean())*100.0)
print("Taux d'erreur sur le jeu de validation : ", (1.0-(validation_predictions==validation_set[:,-1]).mean())*100.0)
