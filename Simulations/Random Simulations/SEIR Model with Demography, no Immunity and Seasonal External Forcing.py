import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

N=150000 #population of village
T=1000 #elapsed time

t=0
beta=1 #human to human transmission rate
epsilon=0.05 #transmission from bats
sigma=0.1 #latency period
mu1=0.1 #recovery rate
mu2=0.1 #disease induced death rate
mu=1/(365*67) #natural birth and death rate
I=1 #intial number of infected individuals
S=N-I #number of susceptibles
E=0 #number of exposed
R=0 #number of recovered
inputSEIR = [S, E, I, R]

SEIR = []
SEIR.append([t, S, E, I, R]) #array of SEIR values

#main loop
while t < T:
    
    if t%365<304 and t%365>120:
        _epsilon=0
                           
    else:
        _epsilon=epsilon
            
    if I==0 and E==0 and _epsilon==0:
        break
    
    N = S + E + I + R
    rate1 = (beta * I * S)/N #human to human transmission
    rate2 = sigma * E #move from exposed to infected
    rate3 = mu1 * I #recovery
    rate4 = mu * N + mu2 * I #birth rate
    rate5 = mu * S #natural death from susceptible class
    rate6 = mu * E #natural death from exposed class
    rate7 = (mu+mu2) * I #natural death from infectious class
    rate8 = _epsilon * S #transmission from bats
    rate9 = mu * R
    ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
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
                    
    elif sum(ratemat [:7])/ratetotal < r and r < sum(ratemat [:8])/ratetotal:
        S = S - 1
        E = E + 1
        
    else:
        R = R - 1
        
    SEIR.append([t, S, E, I, R]) #adds data to SIR matrix
    
time = [row[0] for row in SEIR] #times to be plotted
susceptible = [row[1] for row in SEIR] #susceptible individuals to be plotted
exposed = [row[2] for row in SEIR]
infected = [row[3] for row in SEIR] #infectious individuals to be plotted
recovered = [row[4] for row in SEIR] #dead individuals to be plotted

plt.step(time, exposed, 'b', label='Exposed')
plt.step(time, infected, 'y', label='Infected')
plt.step(time, recovered, 'r', label='Recovered')
plt.step(time, susceptible, 'g', label='Susceptible')
plt.legend(loc='upper right')
plt.xlabel('Time in Days (0 = Jan 1st)')
plt.ylabel('Population')

def dSEIR_dt(X, t):
    return [-beta*X[0]*X[2]/N-epsilon*X[0]+mu*(X[1]+X[2]+X[3])+mu2*X[2], beta*X[0]*X[2]/N+epsilon*X[0]-sigma*X[1]-mu*X[1], sigma*X[1]-X[2]*(mu+mu1+mu2), mu1*X[2]-mu*X[3]]
tsolve=np.linspace(0, T, 1000)

dSEIR = odeint(dSEIR_dt, inputSEIR, tsolve)
plt.plot(tsolve, dSEIR, '--', c='k')


plt.show()