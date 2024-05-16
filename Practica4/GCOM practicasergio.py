# -*- coding: utf-8 -*-

# Sergio Gonzalez Montero
# Victor Martin Martin

import numpy as np
from numpy import cos as cos, sin as sin, pi as pi, sqrt as sqrt
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
import os

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage import io

print("----------------3D SURFACE----------------")
fig = plt.figure(figsize=plt.figaspect(1))

ax = fig.add_subplot(1, 1, 1, projection='3d')

X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
R = -sqrt(X**2 + Y**2)
Z = 3*cos(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.turbo,
                       linewidth=0, antialiased=False)
ax.set_zlim(-10, 10)
plt.title("3D surface")
plt.savefig('3D surface.png', dpi=250)

plt.show()

print("\n----------------ROTATION AND TRANSLATION----------------\n")

def bar3D(x,y,z):
    """
    x, y, z : coordenadas de los puntos del sistema
    Se calcula el baricentro del sistema
    """
    print("Calculating the barycenter...")
    barx, bary, barz = np.mean(x), np.mean(y), np.mean(z)

    return (barx, bary, barz)

def diametro3D(x, y, z):
    """
    x, y, z : coordenadas de los puntos del sistema
    Se calcula el diametro del sistema
    """
    print("\nCalculating the diameter...")
    xaux = x.ravel()
    yaux = y.ravel()
    zaux = z.ravel()
    # Calcula todas las distancias al cuadrado
    distances_squared = (xaux[:, None] - xaux)**2 + \
                        (yaux[:, None] - yaux)**2 + \
                        (zaux[:, None] - zaux)**2
    
    # Ignora las distancias entre el mismo punto
    np.fill_diagonal(distances_squared, 0)
    # Encuentra la distancia maxima al cuadrado
    max_distance_squared = distances_squared.max()
    d = np.sqrt(max_distance_squared)
    
    return d

def trans_iso_afin3D(x, y, z, M, v):
    '''
    x, y, z : coordenadas de los puntos del sistema
    M : matriz de rotacion
    v : vector de traslacion
    '''
    lenx = len(x)
    xt = np.zeros(shape=(lenx,lenx))
    yt = np.zeros(shape=(lenx,lenx))
    zt = np.zeros(shape=(lenx,lenx))
    
    for i in range(len(x)):
        for j in range(len(x)):
            q = np.array([x[i][j], y[i][j], z[i][j]])
            xt[i][j], yt[i][j], zt[i][j] = np.matmul(M,q) + v
            
    return xt, yt, zt

barx, bary, barz = bar3D(X,Y,Z)
print("Barycenter:", round(barx, 3),",", round(bary, 3),",", round(barz, 3))
d = diametro3D(X,Y,Z)
print("Diameter:", round(d, 2))

def animate3D(t):
    """
    Creacion del GIF con paso de frames t
    """
    theta = 3*pi*t
    ro = np.array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta),  0],
                   [0,          0,           1]])
    
    v = np.array([0, 0, d])*t
    print(f"\nt = {round(t,3)}\n ro = \n{ro}\n v = {v}\n")
    ax = plt.axes(xlim=(-8,8), ylim=(-8,8), zlim=(barz-2,d+2),
                  projection='3d')
    
    x, y, z = trans_iso_afin3D(X-barx, Y-bary, Z-barz, ro, v)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.turbo,
                            linewidth=0, antialiased=False)
    return ax,

def init3D():
    return animate3D(0),


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
print("Making the GIF...")
ani = animation.FuncAnimation(fig, animate3D, frames=np.arange(0,1,0.025),
                              init_func=init3D, interval=20)
plt.title(rf"$\rho = 3*\pi$ ; v = [0, 0, d = {round(d, 2)}]")
ani.save("affine isometric transformation.gif", fps = 10)
print("Done\n")


print("----------------HURRICANE ISABEL TRANSFORMATION----------------\n")

def bar2D(x,y):
    """
    x, y : coordenadas de los puntos del sistema
    Se calcula el baricentro del sistema
    """
    print("Calculating the barycenter...")
    barx, bary = np.mean(x), np.mean(y)

    return (barx, bary)

def diametro2D(x, y):
    """
    x, y, z : coordenadas de los puntos del sistema
    Se calcula el diametro del sistema
    """
    print("\nCalculating the diameter...")
    # Envolvente convexa
    points = np.array([x,y]).transpose()
    hull = ConvexHull(points)
    # Extraccion de los puntos de la envolvente
    hullpoints = points[hull.vertices,:]

    # La mayor distancia entre los puntos de la envolvente
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    
    p1 = hullpoints[bestpair[0]]
    p2 = hullpoints[bestpair[1]]
    
    d = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    
    return d

ruta = "D:/2023-2024/SEGUNDO CUATRI/GCOM/Práctica_4"
os.getcwd()
os.chdir(ruta)

img = io.imread("hurricane-isabel.png")

fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,2], cmap=cm.twilight,
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

# Variables de estado coordenadas, azul >= 100
x0 = xx[zz>100]
y0 = yy[zz>100]
z0 = zz[zz>100]/zz.max()
# Variable de estado: color
col = plt.get_cmap("twilight")(np.array(z0))

barx1, bary1 = bar2D(x0, y0)
print("Barycenter:", round(barx1, 3),",", round(bary1, 3))
d1 = diametro2D(x0, y0)
print("Diameter:", round(d1, 3))

def transf2D(x, y, z, M, v):
    '''
    x, y, z : coordenadas de los puntos del sistema
    M : matriz de rotacion
    v : vector de traslacion
    '''
    lenx = len(x)
    xt = np.zeros(lenx)
    yt = np.zeros(lenx)
    zt = np.zeros(lenx)
    
    for i in range(len(x)):
        q = np.array([x[i], y[i], z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v + (barx1, bary1, 0)
        
    return xt, yt, zt

def animate2D(t):
    """
    Creacion del GIF con paso de frames t
    """
    theta = 6*pi*t
    ro = np.array([[cos(theta), -sin(theta), 0],
                   [sin(theta), cos(theta),  0],
                   [0,          0,           1]])
    
    v = np.array([d1, d1, 0]) * t
    print(f"\nt = {round(t, 3)}\n ro = \n{ro}\n v = {v}\n")
    
    ax = plt.axes(xlim=(0, 2.5*d1), ylim=(0, 2.5*d1))

    XYZ = transf2D(x0-barx1, y0-bary1, z0, ro, v)
    
    col = plt.get_cmap("twilight")(np.array(XYZ[2]))
    ax.scatter(XYZ[0], XYZ[1], c=col, s=0.1, animated=True)
    
    return ax,

def init2D():
    return animate2D(0),

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
print("\nMaking the GIF...")
ani = animation.FuncAnimation(fig, animate2D, frames=np.arange(0,1,0.025),
                              init_func=init2D, interval=20)
plt.title(rf"$\rho = 6\cdot\pi$ ; v = [d = {round(d1, 2)}, d = {round(d1, 2)}, 0]")
os.chdir(ruta)
ani.save("hurricane isabel transformation.gif", fps = 10)
print("Done")
os.getcwd()
