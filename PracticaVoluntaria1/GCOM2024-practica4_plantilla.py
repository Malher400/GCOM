# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:58:33 2020

@author: Robert
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from matplotlib import cm
from skimage import io, color
# Importante: El módulo skimage requiere tener instalado:
# la librería scikit-image (por ejemplo con Anaconda o pip)

vuestra_ruta = ""

os.getcwd()
os.chdir(vuestra_ruta)


"""
Ejemplo para el apartado 1.

Modifica la figura 3D y/o cambia el color
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
https://matplotlib.org/stable/gallery/mplot3d/subplot3d.html
"""

fig = plt.figure(figsize=plt.figaspect(0.5))

# plot a 3D surface like in the example mplot3d/surface3d_demo
ax = fig.add_subplot(1, 1, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
R = -np.sqrt(X**2/2 + Y**2/4)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-2.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()


"""
Transformación para el segundo apartado

NOTA: Para el primer aparado es necesario adaptar la función o crear otra similar
pero teniendo en cuenta más dimensiones y hacerla MÁS EFICIENTE
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

vuestra_ruta = ""
os.getcwd()
#os.chdir(vuestra_ruta)

img = io.imread('C:/Users/usu401/Downloads/hurricane-isabel.png')
#dimensions = color.guess_spatial_dimensions(img)
dimensions = img.data.shape
print(dimensions)
io.show()
#io.imsave('hurricane2.png',img)

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
fig = plt.figure(figsize=(5,5))
p = plt.contourf(img[:,:,2],cmap = plt.cm.get_cmap('viridis'), levels=np.arange(100,255,2))
plt.axis('off')
#fig.colorbar(p)

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,2]
z = np.transpose(z)
zz = np.asarray(z).reshape(-1)


"""
Por curiosidad, comparamos el resultado con contourf y scatter!
"""
#Variables de estado coordenadas
x0 = xx[zz>100]
y0 = yy[zz>100]
z0 = zz[zz>100]/zz.max()
#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(z0))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 2, 1)
plt.contourf(x,y,z,cmap = plt.cm.get_cmap('viridis'), levels=np.arange(100,255,2))
ax = fig.add_subplot(1, 2, 2)
plt.scatter(x0,y0,c=col,s=0.1)
plt.show()



def animate(t):
    M = np.array([[1,0,0],[0,1,0],[0,0,1]])
    v=np.array([40,40,0])*t
    
    ax = plt.axes(xlim=(0,800), ylim=(0,800))
    #ax = plt.axes(xlim=(0,800), ylim=(0,800), projection='3d')
    #ax.view_init(60, 30)

    XYZ = transf1D(x0, y0, z0, M=M, v=v)
    col = plt.get_cmap("viridis")(np.array(XYZ[2]))
    ax.scatter(XYZ[0],XYZ[1],c=col,s=0.1,animated=True)
    return ax,

def init():
    return animate(0),

animate(np.arange(0, 1,0.1)[5])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1,0.025), init_func=init,
                              interval=20)
#os.chdir(vuestra_ruta)
ani.save("C:/Users/usu401/Downloads/p4b.gif", fps = 10)  
os.getcwd()





