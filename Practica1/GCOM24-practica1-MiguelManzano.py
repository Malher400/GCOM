# Práctica 1 de Geometría Computacional
# Miguel manzano Rodríguez

import os
import numpy as np
import pandas as pd

#### Vamos al directorio de trabajo
os.getcwd()
#os.chdir(ruta)
#files = os.listdir(ruta)

with open('GCOM2024_pract1_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2024_pract1_auxiliar_esp.txt', 'r',encoding="utf8") as file:
      es = file.read()


#### Contamos cuantos caracteres hay en cada texto
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

#### Transformamos en formato array de los carácteres (states) y su frecuencia
#### Finalmente realizamos un DataFrame con Pandas y ordenamos con 'sort'
tab_en_states = np.array(list(tab_en))
tab_en_weights = np.array(list(tab_en.values()))
tab_en_probab = tab_en_weights/float(np.sum(tab_en_weights))
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index=np.arange(0,len(tab_en_states))

tab_es_states = np.array(list(tab_es))
tab_es_weights = np.array(list(tab_es.values()))
tab_es_probab = tab_es_weights/float(np.sum(tab_es_weights))
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab })
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index=np.arange(0,len(tab_es_states))



##### Para obtener una rama, fusionamos los dos states con menor frecuencia
distr = distr_en
''.join(distr['states'][[0,1]])

### Es decir:
states = np.array(distr['states'])
probab = np.array(distr['probab'])
state_new = np.array([''.join(states[[0,1]])])   #Ojo con: state_new.ndim
probab_new = np.array([np.sum(probab[[0,1]])])   #Ojo con: probab_new.ndim
codigo = np.array([{states[0]: 0, states[1]: 1}])
states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
distr = pd.DataFrame({'states': states, 'probab': probab, })
distr = distr.sort_values(by='probab', ascending=True)
distr.index=np.arange(0,len(states))

#Creamos un diccionario
branch = {'distr':distr, 'codigo':codigo}


## Ahora definimos una función que haga exáctamente lo mismo
def huffman_branch(distr):
    states = np.array(distr['states'])
    probab = np.array(distr['probab'])
    state_new = np.array([''.join(states[[0,1]])])
    probab_new = np.array([np.sum(probab[[0,1]])])
    codigo = np.array([{states[0]: 0, states[1]: 1}])
    states =  np.concatenate((states[np.arange(2,len(states))], state_new), axis=0)
    probab =  np.concatenate((probab[np.arange(2,len(probab))], probab_new), axis=0)
    distr = pd.DataFrame({'states': states, 'probab': probab})
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index=np.arange(0,len(states))
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = np.array([])
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = np.array([branch['codigo']])
        tree = np.concatenate((tree, code), axis=None)
    return(tree)
 
 
tree_en = huffman_tree(distr_en)
tree_en[0].items()
tree_en[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree_en[0].items())[0][1] ## Esto proporciona un '0'
list(tree_en[0].items())[1][1] ## Esto proporciona un '1'

# Igual para la clave en español

tree_es = huffman_tree(distr_es)
tree_es[0].items()
tree_es[0].values()

#Buscar cada estado dentro de cada uno de los dos items
list(tree_es[0].items())[0][1] ## Esto proporciona un '0'
list(tree_es[0].items())[1][1] 



'''
# Apartado i): A partir de las muestras dadas, hallar el código Huffman binario
 de SEng y SEsp, y sus longitudes medias L(SEng) y L(SEsp). Comprobar que se
 satisface el Primer Teorema de Shannon. (1.50 puntos)
'''
        
# Primero, creamos el código huffman de las palabras de un idioma, y lo aplicamos
# para los árboles de los dos idiomas que hemos obtenido. Después, hallamos
# L(SEng) y L(SEsp), además de la entropía.

def codigohuffman(tree):

    lista_arbol = list(tree)
    d = {}
    
    for i in lista_arbol:
        
        [k0, k1] = list(i.keys())
        [v0, v1] = list(i.values())
        
        if (len(k0) == 1):
            d[k0] = str(v0)
        if (len(k1) == 1):
            d[k1] = str(v1)
    
        if (len(k0) > 1):
            for j in k0:
                d[j] = str(v0) + d[j]
        if (len(k1) > 1):
            for j in k1:
                d[j] = str(v1) + d[j]

    return d
        
d_en = codigohuffman(tree_en)
d_es = codigohuffman(tree_es)


def longitud_media(distr, d):
    L_S = 0
    caracteres = list(distr['states'])
    probabilidades = list(distr['probab']) 
    i = 0
    while i < len(caracteres):
        L_S += len(d[caracteres[i]])*probabilidades[i]
        i += 1
    return L_S


def entropia_shannon(distribucion):
    entropia_res = 0
    probabilidades = list(distribucion['probab'])
    
    for probabilidad in probabilidades:
        entropia_res += probabilidad * np.log2(probabilidad)
    
    entropia_res *= -1
    return entropia_res



'''
ii) Utilizando los códigos obtenidos en el apartado anterior, codificar la palabra Lorentz para
ambas lenguas. Comprobar la eficiencia de longitud frente al código binario usual. (0.50 puntos)
'''

def codificacion(palabra, d):
    cod = ''
    for i in palabra:
        cod = cod + d[i]
    return cod

def codificacion_bin(palabra):
    lista = list(format(letra, 'b') for letra in bytearray(palabra, "utf-8"))
    cod = ''
    for i in lista:
        cod = cod + i
    return cod
    

'''
iii) Realiza un programa para decodificar cualquier palabra y comprueba que funciona con el
resultado del apartado anterior para ambos idiomas. (0.50 puntos)
'''

def decodificacion(palabra_codificada, d):
    resultado = ''
    i = 0
    p_temp = ''
    claves = list(d.keys())
    valores = list(d.values())
    
    while i < len(palabra_codificada):
        p_temp += palabra_codificada[i]
        
        if p_temp in valores:
            resultado += claves[valores.index(p_temp)]
            p_temp = ''
            i += 1
        else:
            i += 1
    
    return resultado

