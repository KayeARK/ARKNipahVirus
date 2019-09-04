import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.lines import Line2D

timemat=[]
ifbmat=[]
ifhmat=[]
ifmat=[]
deadmat=[]

for i in range(1):
    
    a=0
    
    while a==0:
   
        N=15000 #population of village
        T=500 #elapsed time

        t=0         
        beta=1/4 #human to human transmission rate
        epsilon=0.001 #transmission from bats
        sigma=1/17 #latency period
        alpha=1 #transmission from hospitals
        omega=1/5 #move to hopsital
        mu1=1/5 #recovery rate
        mu2=1/5 #disease induced death rate
        mu=1/(365*64) #natural birth and death rate
        E=1 #intial number of infected individuals
        S=N-E #number of susceptibles
        I=0 #number of exposed
        R=0 #number of dead
        H=0
        inputSEIHR = [S, E, I, H, R]

        SEIHR = []
        SEIHR.append([t, S, E, I, H, R]) #array of SEIR values
            
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
                
            N = S + E + I + H
            rate1 = (beta * I * S)/N #human to human transmission
            rate2 = (alpha * beta * H * S)/N
            rate3 = _epsilon * S
            rate4 = mu1 * H
            rate5 = mu1 * I
            rate6 = mu * N + mu2 * (I + H)
            rate7 = mu * S
            rate8 = mu * E
            rate9 = mu * I
            rate10 = mu * H
            rate11 = sigma * E
            rate12 = omega * I
            rate13 = mu2 * H
            rate14 = mu2 * I        
                
            ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, rate10, rate11, rate12, rate13, rate14]
            ratetotal = sum(ratemat)
                
            dt = -math.log(random.uniform(0,1))/ratetotal #timestep
            t = t + dt
                
            r=random.uniform(0,1)
                
            if r < sum(ratemat [:1])/ratetotal: #move to infected class
                S = S - 1
                E = E + 1
                ifh = ifh + 1
                a = a + 1
                    
            elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
                S = S - 1
                E = E + 1
                ifh = ifh + 1
                a = a + 1
                    
            elif sum(ratemat [:2])/ratetotal < r and r < sum(ratemat [:3])/ratetotal:
                S = S - 1
                E = E + 1
                ifb = ifb + 1
                a = a + 1
                    
            elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
                H = H - 1
                S = S + 1
                    
            elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
                I = I - 1
                S = S + 1
                    
            elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
                S = S + 1
                    
            elif sum(ratemat [:6])/ratetotal < r and r < sum(ratemat [:7])/ratetotal:
                S = S - 1
                    
            elif sum(ratemat [:7])/ratetotal < r and r < sum(ratemat [:8])/ratetotal:
                E = E - 1
                    
            elif sum(ratemat [:8])/ratetotal < r and r < sum(ratemat [:9])/ratetotal:
                I = I - 1
                
            elif sum(ratemat [:9])/ratetotal < r and r < sum(ratemat [:10])/ratetotal:
                H = H - 1    
            
            elif sum(ratemat [:10])/ratetotal < r and r < sum(ratemat [:11])/ratetotal:
                E = E - 1
                I = I + 1
            
            elif sum(ratemat [:11])/ratetotal < r and r < sum(ratemat [:12])/ratetotal:
                I = I - 1
                H = H + 1
            
            elif sum(ratemat [:12])/ratetotal < r and r < sum(ratemat [:13])/ratetotal:
                H = H - 1
                R = R + 1
                
            else:
                I = I - 1
                R = R + 1    
               
                    
            SEIHR.append([t, S, E, I, H, R]) #adds data to SIR matrix
            
            
            timemat.append(t)
            
        time = [row[0] for row in SEIHR] #times to be plotted
        susceptible = [row[1] for row in SEIHR] #susceptible individuals to be plotted
        exposed = [row[2] for row in SEIHR]
        infected = [row[3] for row in SEIHR] #infectious individuals to be plotted
        hospitalised = [row[4] for row in SEIHR]
        dead = [row[5] for row in SEIHR] #dead individuals to be plotted

        plt.step(time, exposed, 'b', label='Exposed')
        plt.step(time, infected, 'y', label='Infected')
        plt.step(time, dead, 'r', label='Dead')
        plt.step(time, hospitalised, 'm', label='Hospitalised')
        plt.step(time, susceptible, 'g', label='Susceptible')
        plt.xlabel('Time in Days (0 = Jan 1st)')
        plt.ylabel('Population')
    
    ifbmat.append(ifb)
    ifhmat.append(ifh)
    ifmat.append(ifb+ifh)
    deadmat.append(R)
    
print('Average number of total infections is',np.mean(ifmat))
print('Average number of total infections from bats is',np.mean(ifbmat))
print('Average number of total infections from humans is',np.mean(ifhmat))
print('Average number of deaths is',np.mean(deadmat))
print('Standard deviation of number of total infections is',np.std(ifmat))
print('Standard deviation of number of number of deaths is',np.std(deadmat))

colors = ['blue', 'yellow', 'magenta', 'red', 'green']
lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors]
labels = ['Exposed', 'Infectious', 'Hospitalised', 'Dead', 'Susceptible']
plt.legend(lines, labels)


def dSEIHR_dt(X, t):
    return [mu2*(X[2]+X[3])+mu*N+mu1*(X[3]+X[2])-X[0]*((beta*X[2]/N)+(alpha*beta*X[3]/N)+epsilon+mu),X[0]*(epsilon+(alpha*beta*X[3]/N)+(beta*X[2]/N))-X[1]*(mu+sigma) , sigma*X[1]-X[2]*(mu+mu1+mu2+omega),omega*X[2]-X[3]*(mu+mu1+mu2) ,mu2*(X[3]+X[2])]
tsolve=np.linspace(0, T, 1000)

dSEIHR = odeint(dSEIHR_dt, inputSEIHR, tsolve)
plt.plot(tsolve, dSEIHR, '--', c='k')

plt.show()