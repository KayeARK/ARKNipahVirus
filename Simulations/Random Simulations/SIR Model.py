import math
import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt

N=1500 #population of village
T=1000 #elapsed time
t=0 #start time
beta=0.1 #transmission rate
gamma=0.01 #recovery rate
I=1 #intial number of infected individuals
S=N-I
R=0
inputSIR = [S, I, R]

SIR = []
SIR.append([t, S, I, R]) #array of SIR values

#main loop
while t < T:
    
    if I==0:
        break
    
    rate1 = (beta * I * S)/N
    rate2 = gamma * I
    ratetotal = rate1 + rate2
    
    dt = -math.log(random.uniform(0,1))/ratetotal #timestep
    t = t + dt
    
    if random.uniform(0,1) < rate1/ratetotal: #move to infected class
        S = S - 1
        I = I + 1
        
    else: #move to recovered class
        I = I - 1
        R = R + 1
        
    SIR.append([t, S, I, R]) #adds data to SIR matrix
    
time = [row[0] for row in SIR] #times to be plotted
susceptible = [row[1] for row in SIR] #susceptible individuals to be plotted
infected = [row[2] for row in SIR] #infectious individuals to be plotted
recovered = [row[3] for row in SIR] #recovered individuals to be plotted

plt.plot(time, susceptible, 'g', label='Susceptible')
plt.plot(time, infected, 'r', label='Infected')
plt.plot(time, recovered, 'b', label='Recovered')
plt.legend(loc='center right')
plt.xlabel('Time')
plt.ylabel('Population')

def dSIR_dt(X, t):
    return [-beta*X[0]*X[1]/N, beta*X[0]*X[1]/N - gamma*X[1], gamma*X[1]]
tsolve=np.linspace(0, T, 1000)

dSIR = odeint(dSIR_dt, inputSIR, tsolve)
plt.plot(tsolve, dSIR, '--', c='k')

plt.show()