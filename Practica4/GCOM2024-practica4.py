# -*- coding: utf-8 -*-
"""

@author: Miguel Manzano Rodríguez
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from matplotlib import cm
from skimage import io, color
import math
# Importante: El módulo skimage requiere tener instalado:
# la librería scikit-image (por ejemplo con Anaconda o pip)

from scipy.spatial import ConvexHull

vuestra_ruta = ""

# FIGURA 1 DE EJEMPLO

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 1, 1, projection='3d')

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
R = -np.sqrt(X**2/2 + Y**2/4)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.afmhot,
                       linewidth=0, antialiased=False)
ax.set_zlim(-2.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()


"""
#################################################################
APARTADO 1: Familia paramétrica continua: rotación y traslación.
#################################################################
"""


xf = X.flatten()
yf = Y.flatten()
zf = Z.flatten()


centroide = np.array([np.mean(X), np.mean(Y), np.mean(Z)])

K = np.array([[X[i][j], Y[i][j], Z[i][j]] for j in range(len(X[0])) for i in range(len(X))])

hull = ConvexHull(K)

distances = np.linalg.norm([(X[i] - X[j], Y[i] - Y[j], Z[i] - Z[j]) 
                            for i in range(len(X)) for j in range(len(X)) if i != j], axis=1)
diametro = distances.max()


print("Centroide: ", centroide, "\nDiametro: ", round(diametro, 3));

fig = plt.figure(figsize=(6,6));

m0 = len(X[0])
n0 = len(X)

def transformacionIso(x,y,z, M, v):

    xt = x * 0
    yt = y * 0
    zt = z * 0
    
    for i in range(len(x)):
        q = np.array([x[i], y[i], z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
        
    return xt, yt, zt



def animate(t):

    theta = 3*math.pi*t
    
    M = np.array([[np.cos(theta), -np.sin(theta), 0], 
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    
    v = np.array([0, 0, diametro * t])
    
    (xt, yt, zt) = transformacionIso(xf-centroide[0], yf-centroide[1],
                                     zf-centroide[2], M, v)

    X = np.array([[xt[m0 * i + j] for j in range(m0)] for i in range(n0)])
    Y = np.array([[yt[m0 * i + j] for j in range(m0)] for i in range(n0)])
    Z = np.array([[zt[m0 * i + j] for j in range(m0)] for i in range(n0)])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.afmhot,
                            linewidth=0, antialiased=False)
    return ax,

def init():
    return animate(0),

'''
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1,0.1),
                              init_func=init, interval=20)
ani.save("p4a.gif", fps = 10)
'''

            
"""
###############################################################
APARTADO 2: Subsistema de azul y rotación y traslado de nuevo.
###############################################################
"""

def transf1D(x,y,z,M, v=np.array([0,0,0])):
    xt = x*0
    yt = x*0
    zt = x*0
    for i in range(len(x)):
        q = np.array([x[i],y[i],z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt

"""
Segundo apartado casi regalado

https://education.nationalgeographic.org/resource/coriolis-effect/
"""

img = io.imread('hurricane-isabel.png')

dimensions = img.data.shape
print(dimensions)
io.show()

fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,2],cmap = plt.cm.get_cmap('afmhot'),
                 levels=np.arange(100,255,2))
plt.axis('off')

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,2]
z = np.transpose(z)
zz = np.asarray(z).reshape(-1)

x0 = xx[zz>=100]
y0 = yy[zz>=100]
z0 = zz[zz>=100]/zz.max()

centroide = np.array([np.mean(x0), np.mean(y0), np.mean(z0)])

print(centroide)

X = np.array([x0,y0,z0]).T
hull = ConvexHull(X)

diametro = 0

for i in hull.vertices:
    for j in hull.vertices:
        if math.dist(X[i], X[j]) > diametro:
            diametro = math.dist(X[i], X[j])

print("Centroide: ", centroide, "\nDiametro: ", diametro);


def animate2(t):
    theta = 6 * math.pi * t;
    M = np.array([[np.cos(theta), -np.sin(theta), 0], 
                  [np.sin(theta), np.cos(theta), 0], 
                  [0, 0, 1]])
    v=np.array([diametro,diametro,0])*t
    
    print(t)
    
    ax = plt.axes(xlim=(0,1200), ylim=(0,1200))

    XYZ = transf1D(x0, y0, z0, np.array([[1,0,0], [0,1,0], [0,0,1]]),
                   v=centroide*(-1))
    XYZ = transf1D(XYZ[0], XYZ[1], XYZ[2], M=M, v = centroide)
    XYZ = transf1D(XYZ[0], XYZ[1], XYZ[2],np.array([[1,0,0], [0,1,0],
                                                    [0,0,1]]), v=v)
    
    col = plt.get_cmap("afmhot")(np.array(0.1+z0))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init2():
    return animate2(0),

'''
fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate2, frames=np.arange(0,1,0.1),
                              init_func=init2, interval=20)

ani.save("p4b.gif", fps = 10)  
'''


