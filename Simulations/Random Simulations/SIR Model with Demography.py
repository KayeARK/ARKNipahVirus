import math
import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt

N=15000 #population of village
T=1000 #elapsed time
t=0 #start time
beta=0.1 #transmission rate
gamma=0.01 #recovery rate
mu=1/(365*80) #natural death rate
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
    rate3 = mu * S
    rate4 = mu * I
    rate5 = mu * R
    ratetotal = rate1 + rate2 + rate3 + rate4 + rate5
    
    dt = -math.log(random.uniform(0,1))/ratetotal #timestep
    t = t + dt
    
    r = random.uniform(0,1)
    
    if r < rate1/ratetotal: #move to infected class
        S = S - 1
        I = I + 1
        
    elif rate1/ratetotal < r and r < (rate1 + rate2)/ratetotal: #move to recovered class
        I = I - 1
        R = R + 1
        
    elif (rate1 + rate2)/ratetotal < r and r < (rate1 + rate2 + rate3)/ratetotal:
        S = S
        
    elif (rate1 + rate2 + rate3)/ratetotal < r and r < (rate1 + rate2 + rate3 + rate4)/ratetotal:
        I = I - 1
        S = S + 1
        
    else:
        R = R - 1
        S = S + 1
        

        
    SIR.append([t, S, I, R]) #adds data to SIR matrix
    
time = [row[0] for row in SIR] #times to be plotted
susceptible = [row[1] for row in SIR] #susceptible individuals to be plotted
infected = [row[2] for row in SIR] #infectious individuals to be plotted
recovered = [row[3] for row in SIR] #recovered individuals to be plotted

plt.plot(time, susceptible, 'g', label='Susceptible')
plt.plot(time, infected, 'r', label='Infected')
plt.plot(time, recovered, 'b', label='Recovered')
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Population')

def dSIR_dt(X, t):
    return [mu*N - beta*X[0]*X[1]/N - mu*X[0], beta*X[0]*X[1]/N - gamma*X[1] - mu*X[1], gamma*X[1] - mu*X[2]]
tsolve=np.linspace(0, T, 1000)

dSIR = odeint(dSIR_dt, inputSIR, tsolve)
plt.plot(tsolve, dSIR, '--', c='k')

plt.show()