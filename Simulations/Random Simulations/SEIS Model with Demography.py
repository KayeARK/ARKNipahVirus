import math
import numpy as np
from scipy.integrate import odeint
import random
import matplotlib.pyplot as plt

N=10000 #population of village
T=1000 #elapsed time
t=0 #start time
beta=0.1 #transmission rate
sigma=1/14 #rate of movement from exposed to infected
mu1=0.004 #recovery rate
mu2=0.001 #disease induced death rate
mu=1/(365*80) #natural birth and death rate
I=1 #intial number of infected individuals
S=N-I #number of susceptibles
E=0 #number of exposed
inputSEIS = [S, E, I]

SEIS = []
SEIS.append([t, S, E, I]) #array of SEIR values

#main loop
while t < T:
    
    if I==0 and E==0:
        break
    
    N = S + E + I
    rate1 = (beta * I * S)/N #disease transmission
    rate2 = sigma * E #move from latency period to infectious class
    rate3 = mu1 * I #recovery and transfer back to susceptible class
    rate4 = mu * N #birth rate
    rate5 = mu * S #natural death rate from susceptible class
    rate6 = mu * E #natural death rate from exposed class
    rate7 = mu * I #natural death rate from infectious class
    ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7]
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
        S = S + 1
        
    elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
        S = S + 1
        
    elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
        S = S - 1
        
    elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
        E = E - 1        
            
    else:
        I = I - 1
        
    SEIS.append([t, S, E, I]) #adds data to SIR matrix
    
time = [row[0] for row in SEIS] #times to be plotted
susceptible = [row[1] for row in SEIS] #susceptible individuals to be plotted
exposed = [row[2] for row in SEIS]
infected = [row[3] for row in SEIS] #infectious individuals to be plotted

plt.plot(time, susceptible, 'g', label='Susceptible')
plt.plot(time, exposed, 'y', label='Exposed')
plt.plot(time, infected, 'r', label='Infected')
plt.legend(loc='center right')
plt.xlabel('Time')
plt.ylabel('Population')

def dSEIS_dt(X, t):
    return [-beta*X[0]*X[2]/N +mu1*X[2] + mu*(N-X[0]), beta*X[0]*X[2]/N - sigma*X[1] - mu*X[1], (sigma*X[1])-X[2]*(mu1+mu)]
tsolve=np.linspace(0, T, 1000)

dSEIS = odeint(dSEIS_dt, inputSEIS, tsolve)
plt.plot(tsolve, dSEIS, '--', c='k')

plt.show()