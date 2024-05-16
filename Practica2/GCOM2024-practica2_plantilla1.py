# -*- coding: utf-8 -*-
"""
Plantilla 1 de la práctica 2

Referencia: 
    https://scikit-learn.org/stable/modules/clustering.html
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
#from scipy.spatial import ConvexHull, convex_hull_plot_2d
#from scipy.spatial import Voronoi, voronoi_plot_2d


# #############################################################################
# Aquí tenemos definido el sistema X de 1500 elementos (personas) con dos estados
archivo1 = "C:/Users/rober/Downloads/Personas_de_villa_laminera.txt"
archivo2 = "C:/Users/rober/Downloads/Franjas_de_edad.txt"
X = np.loadtxt(archivo1,skiprows=1)
Y = np.loadtxt(archivo2,skiprows=1)
labels_true = Y[:,0]

header = open(archivo1).readline()
print(header)
print(X)
#Si quisieramos estandarizar los valores del sistema, haríamos:
#from sklearn.preprocessing import StandardScaler
#X = StandardScaler().fit_transform(X)  

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

# #############################################################################
# Los clasificamos mediante el algoritmo KMeans
n_clusters=2

#Usamos la inicialización aleatoria "random_state=0"
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_
silhouette = metrics.silhouette_score(X, labels)

# Etiqueta de cada elemento (punto)
print(kmeans.labels_)
# Índice de los centros de vencindades o regiones de Voronoi para cada elemento (punto) 
print(kmeans.cluster_centers_)
#Coeficiente de Silhouette
print("Silhouette Coefficient: %0.3f" % silhouette)


# #############################################################################
# Predicción de elementos para pertenecer a una clase:
problem = np.array([[-1.5, -1], [1.5, -1]])
clases_pred = kmeans.predict(problem)
print(clases_pred)

# #############################################################################
# Representamos el resultado con un plot

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(8,4))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.plot(problem[:,0],problem[:,1],'o', markersize=12, markerfacecolor="red")

plt.title('Fixed number of KMeans clusters: %d' % n_clusters)
plt.show()


