# -*- coding: utf-8 -*-
"""
@author: Miguel Manzano Rodríguez
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import sys

# Carga de datos
archivo1 = "Personas_de_villa_laminera.txt"
archivo2 = "Franjas_de_edad.txt"
X = np.loadtxt(archivo1, skiprows=1)
Y = np.loadtxt(archivo2, skiprows=1)
labels_true = Y[:,0]

plt.plot(X[:,0],X[:,1],'ro', markersize=1)
plt.show()

# FUNCIÓN PARA DIBUJAR UNA GRÁFICA
def plot_generica(X,Y,xlabel,ylabel,title):
    plt.figure(figsize=(8,4))
    plt.plot(X, Y, color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    
##############################################################################
'''
i) Obtén el coeficiente s de A para diferente número de vecindades 
   k ∈ {2, 3, ..., 15} usando el algoritmo KMeans. Muestra en una gráfica el
   valor de s en función de k y decide con ello cuál es el número óptimo de
   vecindades. En una segunda gráfica, muestra la clasificación (clusters)
   resultante con diferentes colores y representa el diagrama de Voronói en
   esa misma gráfica.
'''
# #############################################################################

# Los clasificamos mediante el algoritmo KMeans
n_clusters=range(2,16,1)

# Obtenemos el coeficiente de Silhoutte de X para las diferentes vecindades.
# Utilizamos la inicialización aleatoria "random_state=0"

silhouette = list()
for k in n_clusters:
    k_means = kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    silhouette.append(metrics.silhouette_score(X, labels))

# Ahora, creamos una gráfica con el valor de Silhouette en función de k.

plot_generica(n_clusters, silhouette, 'n_clusters', 'silhouette',
              'Valor de silhouette en función del número de clusters')

# Obtenemos el número óptimo de vecindades
vecindades_opt = n_clusters[np.argmax(silhouette)]
    
# Obtenemos la clasificación para ese número de vecindades
kmeans = KMeans(n_clusters=vecindades_opt, random_state=0).fit(X)


# Representamos el resultado con un plot

unique_labels = set(kmeans.labels_)
colors = [plt.cm.Spectral(each) for each
          in np.linspace(0, 1, len(unique_labels))]

voronoi_plot_2d(Voronoi(kmeans.cluster_centers_), show_vertices = False)

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (kmeans.labels_ == k)

    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

plt.autoscale()
plt.title('Número fijo de clusters de KMeans: %d' % vecindades_opt)
plt.show()

###############################################################################
'''
ii) Obtén el coeficiente s para el mismo sistema A usando ahora el algoritmo
    DBSCAN con la métrica 'euclidean' y luego con 'manhattan'. En este caso,
    el parámetro que debemos explorar es el umbral de distancia
    ϵ ∈ (0.1, 0.4), fijando el número de elementos mínimo en n0 = 10.
    Comparad gráficamente con el resultado del apartado anterior.
'''
###############################################################################

epsilon_rg = np.arange(0.1, 0.4, 0.001)

silhouette_euclidean = list()
n_clusters_euclidean = list()

for epsilon in epsilon_rg:
    db = DBSCAN(eps=epsilon, min_samples=10, metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_euclidean.append(len(set(labels)) - (1 if -1 in labels else 0))
    silhouette_euclidean.append(metrics.silhouette_score(X,labels))
    n_noise_ = list(labels).count(-1)

# En función de epsilon, mostremos en una gráfica el valor de silhouette

plot_generica(epsilon_rg, silhouette_euclidean, 'epsilon', 'silhouette',
              'Valor de silhouette respecto a épsilon con métrica euclidiana')

dict = {k : -sys.maxsize for k in sorted(n_clusters_euclidean)}

for i in range(len(silhouette_euclidean)):
    dict[n_clusters_euclidean[i]] = max(dict[n_clusters_euclidean[i]], silhouette_euclidean[i])
    
n_clusters_euclidean, silhouette_euclidean = list(dict.keys()), list(dict.values())



silhouette_manhattan = list()
n_clusters_manhattan = list()

for epsilon in epsilon_rg:
    db_m = DBSCAN(eps=epsilon, min_samples=10, metric='manhattan').fit(X)
    core_samples_mask = np.zeros_like(db_m.labels_, dtype=bool)
    core_samples_mask[db_m.core_sample_indices_] = True
    labels = db_m.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_manhattan.append(len(set(labels)) - (1 if -1 in labels else 0))
    silhouette_manhattan.append(metrics.silhouette_score(X,labels))
    n_noise_ = list(labels).count(-1)

# En función de epsilon, mostremos en una gráfica el valor de silhouette

plot_generica(epsilon_rg, silhouette_manhattan, 'epsilon', 'silhouette',
              'Valor de silhouette respecto a épsilon con métrica Manhattan')

dict = {k : -sys.maxsize for k in sorted(n_clusters_manhattan)}


for i in range(len(silhouette_manhattan)):
    dict[n_clusters_manhattan[i]] = max(dict[n_clusters_manhattan[i]], silhouette_manhattan[i])
    
n_clusters_manhattan, silhouette_manhattan = list(dict.keys()), list(dict.values())    
    

plt.figure(figsize=(8,4))
plt.plot(n_clusters, silhouette, color='r', label='KMeans')
plt.plot(n_clusters_euclidean, silhouette_euclidean, color='g', label='DBSCAN(euclidean)')
plt.plot(n_clusters_manhattan, silhouette_manhattan, color='b', label='DBSCAN(manhattan)')
plt.xlabel('n_clusters')
plt.ylabel('s')
plt.legend()
plt.title('Comparación gráfica de los algoritmos con los coeficientes de silhouette')



###############################################################################
'''
iii) ¿De qué franja de edad diríamos que son las personas con coordenadas
    a := (1/2, 0) y b := (0, -3)? Comprueba tu respuesta con la función
    kmeans.predict.
'''

# Número de franjas
n_franjas = max(Y[:,0])

# Cluster al que pertenecen según kmeans.predict
respuesta = kmeans.predict([[0.5, 0], [0, -3]])
a_cluster, b_cluster = respuesta[0], respuesta[1]

# Vemos qué franja es la predominante (la que más se cuente) en esos clusters
clusters_Y = kmeans.predict(Y[:,1:3])

a_count = list();
b_count = list();
for franja in range(int(n_franjas)):
    lista_aux_a = list()
    for i in range(len(Y)):
        lista_aux_a.append((Y[i][0], clusters_Y[i]))
    a_count.append(lista_aux_a.count((franja,a_cluster)))
    b_count.append(lista_aux_a.count((franja,b_cluster)))


print("La franja de la persona a:=(0.5,0) debería ser:", np.argmax(a_count))
print("La franja de la persona b:=(0,-3) debería ser:", np.argmax(b_count))






















