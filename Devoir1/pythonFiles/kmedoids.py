# Auteur : Lenny SIEMENI
# RAPPEL : MNIST dans Scikitlearn a :
	# 10 classes
	# Chaque image est de taille 8*8
	# 1797 echantillons d'images au total
	# donc 1797*64
	# ~ 180 echantillons par classe
	
# Pour avoir le jeu de donnees MNIST au complet
# Telecharger a partir de open_ml
# Ne pas oublier d'ajuster les bornes de partitionnement

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn import datasets
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score

image_size = 28 # longueur et largeur de l image
labels = 10 #  0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
mnist = datasets.load_digits()
indexes = np.arange(mnist.data.shape[0])

train_set = mnist.data[indexes[:700]]
test_set = mnist.data[indexes[700:900]]
validation_set = mnist.data[indexes[900:]]

# Approche similaire a laplace smoothing, a voir si la valeur du facteur est
# pertinante
smoothing_factor = 0.99 / 255
indexes = np.arange(mnist.data.shape[0])


train_set = mnist.data[indexes[:70]] * smoothing_factor + 0.001
test_set = mnist.data[indexes[70:90]] * smoothing_factor + 0.001
validation_set = mnist.data[indexes[90:]] * smoothing_factor + 0.001

#train_labels = mnist.data[train_set[:,:1] * smoothing_factor + 0.001]
#test_labels = mnist.data[test_set[:,:1]]
#validation_labels = mnist.data[validation_set[:,:1]]
mnist = mnist.data # on selectionne toutes les colonnes mais pas les labels
print(mnist)
#
## turn it into a dataframe 
#d = pd.DataFrame(mnist)
# 
## plot the data 
#plt.scatter(d[0], d[1]);
##plt.scatter(d[:,0], d[:,1]);
#
##gmm = GaussianMixture(n_components = 10)
##gmm.fit(d) 
##
### Assign a label to each sample 
##labels = gmm.predict(d) 
##d['labels']= labels 
##d0 = d[d['labels']== 0] 
##d1 = d[d['labels']== 1] 
##d2 = d[d['labels']== 2] 
##
### plot three clusters in same plot 
##plt.scatter(d0[0], d0[1], c ='r') 
##plt.scatter(d1[0], d1[1], c ='yellow') 
##plt.scatter(d2[0], d2[1], c ='g') 

plt.show()


