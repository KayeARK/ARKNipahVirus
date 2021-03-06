import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

a=0

while a==0:

    N=1500 #population of village
    T=500 #elapsed time
        
    t=0  
    beta=17 #human to human transmission rate
    epsilon=0.00006 #transmission from bats
    sigma=0.1 #latency period
    mu1=1/5 #recovery rate
    mu2=1/5 #disease induced death rate
    mu=1/(365*64) #natural birth and death rate
    I=1 #intial number of infected individuals
    S=N-I #number of susceptibles
    E=0 #number of exposed
    R=0 #number of dead
    inputSEIR = [S, E, I, R]

    SEIR = []
    SEIR.append([t, S, E, I, R]) #array of SEIR values
    
    ifb=0 #infection from bats
    ifh=0 #infection from human to human

    #main loop
    while t < T:
        
        if t%365<304 and t%365>120:
            _epsilon=0
                           
        else:
            _epsilon=epsilon
            
        if I==0 and E==0 and _epsilon==0:
            break
        
        N = S + E + I
        rate1 = (beta * I * S)/N #human to human transmission
        rate2 = sigma * E #move from exposed to infected
        rate3 = mu2 * I #disease induced death
        rate4 = mu1 * I #recovery from the disease and move back to susceptible
        rate5 = mu * N #birth rate
        rate6 = mu * S #natural death from susceptible class
        rate7 = mu * E #natural death from exposed class
        rate8 = mu * I #natural death from infectious class
        rate9 = _epsilon * S #transmission from bats
        ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        ratetotal = sum(ratemat)
        
        dt = -math.log(random.uniform(0,1))/ratetotal #timestep
        t = t + dt
        
        r=random.uniform(0,1)
        
        if r < sum(ratemat [:1])/ratetotal: #move to infected class
            S = S - 1
            E = E + 1
            a = a + 1
            ifh = ifh + 1
            
        elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
            E = E - 1
            I = I + 1
            
        elif sum(ratemat [:2])/ratetotal < r and r < sum(ratemat [:3])/ratetotal:
            I = I - 1
            R = R + 1
            
        elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
            I = I - 1
            S = S + 1
            
        elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
            S = S + 1
            
        elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
            S = S - 1
            
        elif sum(ratemat [:6])/ratetotal < r and r < sum(ratemat [:7])/ratetotal:
            E = E - 1
            
        elif sum(ratemat [:7])/ratetotal < r and r < sum(ratemat [:8])/ratetotal:
            I = I - 1
                             
        else:
            S = S - 1
            E = E + 1
            a = a + 1
            ifb = ifb + 1
            
        SEIR.append([t, S, E, I, R]) #adds data to SIR matrix
    
time = [row[0] for row in SEIR] #times to be plotted
susceptible = [row[1] for row in SEIR] #susceptible individuals to be plotted
exposed = [row[2] for row in SEIR]
infected = [row[3] for row in SEIR] #infectious individuals to be plotted
dead = [row[4] for row in SEIR] #dead individuals to be plotted

plt.step(time, exposed, 'b', label='Exposed')
plt.step(time, infected, 'y', label='Infected')
plt.step(time, dead, 'r', label='Dead')
plt.legend(loc='upper right')
plt.xlabel('Time in Days (0 = Jan 1st)')
plt.ylabel('Population')

print('Number of infections from bats is', ifb)
print('Number of infections from humans is', ifh)



plt.show()