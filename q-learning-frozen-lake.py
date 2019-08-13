from __future__ import print_function
import gym
import time
import numpy as np

""""
Este ambiente consta de una matriz 4x4 que representa un lago.
Los estados y las accciones son discretas, el objetivo: Que el agente
aprenda a navegar de principio a fin sin caer al hueco.
"""

#cargamos el entorno de la libreria
env = gym.make('FrozenLake-v0')
#reiniciamos el estado del juego
s = env.reset()
#print("estado inicial es:", s)
##con el comando .render podemos visualizar el juego
#env.render()

def epsilon_codicioso(tabla_Q, s, na):
    eps = 0.3
    p = np.random.uniform(low= 0, high= 1)
    if p > eps:
        return np.argmax(tabla_Q[s, :]) #  para cada estado considere la acción que tiene el valor Q más alto
    else:
        return env.action_space.sample()
#-----------------------    
#implementación Q-learning

#creamos la tabla-Q
tabla_Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.5
gamma = 0.9
episodios = 100000

for i in range(episodios):
    s = env.reset()
    done = False
    while True:
        a = epsilon_codicioso(tabla_Q, s, env.action_space.n)
        prox_s, r, done, info = env.step(a)
        if r == 0:
            if done == True:
                r = -5 #para dar recompensas negativas cuando aparecen agujeros
                tabla_Q[prox_s] = np.ones(env.action_space.n)*r
            else:
                r = -1 #para dar recompensas negativas para anular rutas largas
        if r == 1:
            r = 100
            tabla_Q[prox_s] = np.ones(env.action_space.n)*r
            
        tabla_Q[s, a] = tabla_Q[s, a] + alpha*(r + gamma*np.max(tabla_Q[prox_s,a]) - tabla_Q[s,a])
        s = prox_s
        if done == True:
            break
print("Tabla Q:")
print(tabla_Q)
print("salida despues del aprendizaje")
#verifiquemos cuánto ha aprendido nuestro agente
s = env.reset()
env.render()
while True:
    a = np.argmax(tabla_Q[s])
    prox_s, r, done, info = env.step(a)
    print("=========")
    env.render()
    s = prox_s
    if done == True:
        break
