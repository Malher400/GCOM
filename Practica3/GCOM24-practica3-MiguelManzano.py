'''
Práctica 3 - Miguel Manzano Rodríguez
'''

import os #?
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

os.getcwd()  #?

#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal

def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq

#Ecuación de un sistema dinámico continuo
#Ejemplo de oscilador simple
def F(q):
    ddq = - 2*q*(q**2-1)
    return ddq

#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
#Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),

def periodos(q,d,max=True):
    #Si max = True, tomamos las ondas a partir de los máximos/picos
    #Si max == False, tomamos las ondas a partir de los mínimos/valles
    epsilon = 5*d
    dq = deriv(q,dq0=None,d=d) #La primera derivada es irrelevante
    if max == True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q >0))
    if max != True:
        waves = np.where((np.round(dq,int(-np.log10(epsilon))) == 0) & (q <0))
    diff_waves = np.diff(waves)
    waves = waves[0][1:][diff_waves[0]>1]
    pers = diff_waves[diff_waves>1]*d
    return pers, waves

#################################################################    
#  CÁLCULO DE ÓRBITAS
################################################################# 

q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(5,5))
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([3,4])
horiz = 32
for i in iseq:
    d = 1./10**i
    n = int(horiz/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+str(np.around(d,4)),c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)

#Ejemplo de coordenadas canónicas (q, p)
#Nos quedamos con el más fino y calculamos la coordenada canónica 'p'
d = 1./10**4
n = int(horiz/d)
t = np.arange(n+1)*d
q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
dq = deriv(q, dq0=dq0, d=d)
p = dq/2 # HACER FUNCIÓN TODO ESTO PUES LO REPITO ABAJO

#Ejemplo gráfico de la derivada de q(t)
fig, ax = plt.subplots(figsize=(12,5))
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("dq(t)", fontsize=12)
plt.plot(t, dq, '-')

#Ejemplo de diagrama de fases (q, p) para una órbita completa
fig, ax = plt.subplots(figsize=(5,5))
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.plot(q, p, '-')
plt.show()

#################################################################    
#  ESPACIO FÁSICO
################################################################# 


## Pintamos el espacio de fases
def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col))


Horiz = 12
d = 10**(-4)


fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = fig.add_subplot(1,1, 1)
#Condiciones iniciales:
seq_q0 = np.linspace(0.,1.,num=10)
seq_dq0 = np.linspace(0.,2,num=10)
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='ro',d= 10**(-4),n = int(Horiz/d))
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.show()



'''
APARTADO II. Obtén el valor del área de D_t para t = 1/3 y una estimación de
su intervalo de error.
'''

# Función para calcular el área de D_t
def area (t, d, N): # Parámetros t, delta, 
        
    fig, ax = plt.subplots(figsize=(8, 5))  # Creamos la figura y el eje una vez
    
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)


    #Condiciones iniciales:
    seq_q0 = np.linspace(0,1, N)
    seq_dq0 = np.linspace(0,2, N)
    q2 = np.array([])
    p2 = np.array([])
    for i in range(N):
        for j in range(N):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            n = int(t/d)
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            q2 = np.append(q2, q[-1])
            p2 = np.append(p2, p[-1])
            
            plt.xlim(-2.2, 2.2)
            plt.ylim(-1.2, 1.2)

            
            plt.rcParams["legend.markerscale"] = 6
            ax.set_xlabel("q(t)", fontsize=12)
            ax.set_ylabel("p(t)", fontsize=12)
            ax.plot(q[-1], p[-1], marker="o", markersize = 2, markeredgecolor="red", markerfacecolor="red")
            
    
    X = np.array([q2, p2]).T
    hull = ConvexHull(X)
    print("Área:", hull.volume)
    return hull.volume
    plt.show()


deltas = np.linspace(3, 4, num=5)
areas = [area(1/3, 10**(-d), int(10*d)) for d in deltas]
resta_areas = [abs(areas[i]-areas[4]) for i in range(len(areas)-1)]
sort_resta_areas = sorted(resta_areas)
print("El área es aprox.", round(areas[4], 3), "con", round(sort_resta_areas[3], 3), "de error.")



'''
APARTADO III. Animación GIF
'''

from matplotlib import animation


def animate(t):
    ax = plt.axes()
    ax.clear()
    d = 10**(-4)
    if (t == 0):
        t += 0.001
    n = int(t/d)
    N = 100    
    seq_q0 = np.linspace (0,1, num=N)
    seq_dq0 = np.linspace (0,2,num=N)
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                q0 = seq_q0[i]
                dq0 = seq_dq0[j]
                q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
                dq = deriv(q, dq0=dq0, d=d)
                p = dq/2
                plt.xlim(-2.2, 2.2)
                plt.ylim(-1.2, 1.2)
                plt.rcParams["legend.markerscale"] = 6
                ax.set_xlabel("q(t)", fontsize=12)
                ax.set_ylabel("p(t)", fontsize=12)
                ax.plot(q[-1], p[-1], marker="o", markersize = 2, markeredgecolor="red", markerfacecolor="red")
    return ax,

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, func=animate, frames=np.arange(0,5,0.3), interval=20)
ani.save("ejemplo.gif", fps = 5)
