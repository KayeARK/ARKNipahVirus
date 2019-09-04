import math
import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt

N=1000 #population of village
T=1000 #elapsed time
t=0 #start time
beta=0.1 #transmission rate
sigma=1/14 #rate of movement from exposed to infected
gamma=0.005 #recovery rate
mu=1/(365*80) #natural birth and death rate
I=1 #intial number of infected individuals
S=N-I
R=0
E=0
inputSEIR = [S, E, I, R]

SEIR = []
SEIR.append([t, S, E, I, R]) #array of SEIR values

#main loop
while t < T:
    
    if I==0:
        break
    
    N = S + E + I + R
    rate1 = (beta * I * S)/N
    rate2 = sigma * E
    rate3 = gamma * I
    rate4 = mu * N
    rate5 = mu * S
    rate6 = mu * E
    rate7 = mu * I
    rate8 = mu * R
    ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8]
    ratetotal = sum(ratemat)  
    dt = -math.log(random.uniform(0,1))/ratetotal #timestep
    t = t + dt
    
    r=random.uniform(0,1)
    
    if r < sum(ratemat [:1])/ratetotal: #move to infected class
        S = S - 1
        E = E + 1
        
    elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
        E = E - 1
        I = I + 1
        
    elif sum(ratemat [:2])/ratetotal < r and r < sum(ratemat [:3])/ratetotal:
        I = I - 1
        R = R + 1
        
    elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
        S = S + 1
        
    elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
        S = S - 1
        
    elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
        E = E - 1
        
    elif sum(ratemat [:6])/ratetotal < r and r < sum(ratemat [:7])/ratetotal:
        I = I - 1
             
    else:
        R = R - 1
        
    SEIR.append([t, S, E, I, R]) #adds data to SIR matrix
    
time = [row[0] for row in SEIR] #times to be plotted
susceptible = [row[1] for row in SEIR] #susceptible individuals to be plotted
exposed = [row[2] for row in SEIR]
infected = [row[3] for row in SEIR] #infectious individuals to be plotted
recovered = [row[4] for row in SEIR] #recovered individuals to be plotted

plt.plot(time, susceptible, 'g', label='Susceptible')
plt.plot(time, exposed, 'y', label='Exposed')
plt.plot(time, infected, 'r', label='Infected')
plt.plot(time, recovered, 'b', label='Recovered')
plt.legend(loc='center right')
plt.xlabel('Time')
plt.ylabel('Population')

def dSEIR_dt(X, t):
    return [mu*N -beta*X[0]*X[2]/N - mu*X[0], beta*X[0]*X[2]/N - sigma*X[1] - mu*X[1], sigma*X[1]-gamma*X[2] - mu*X[2] , gamma*X[2] - mu*X[3]]
tsolve=np.linspace(0, T, 1000)

dSEIR = odeint(dSEIR_dt, inputSEIR, tsolve)
plt.plot(tsolve, dSEIR, '--', c='k')

plt.show()