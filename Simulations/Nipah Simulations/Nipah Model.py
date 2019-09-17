import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.lines import Line2D

ifbmat=[]
ifhmat=[]
ifmat=[]
deadmat=[]


for i in range(1000):
    
    a=0
    
    while a==0:        
        
        N=1500 #population of village
        T=365 #elapsed time

        t=50  

        data = np.loadtxt('Faridpur2004ABCdata.dat')

        betamat=data[:,98]
        epsilonmat=data[:,99]
        sigmamat=data[:,100]
        seasonstartmat=data[:,101]
        seasonendmat=data[:,102]
        msemat=data[:,103]
        Emat=data[:,104]
        
        histbeta,binsbeta = np.histogram(betamat)
        bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
        cdfbeta=np.cumsum(histbeta)
        cdfbeta=cdfbeta / cdfbeta[-1]
        valuesbeta = np.random.rand(1)
        value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
        beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

        histepsilon,binsepsilon = np.histogram(epsilonmat)
        bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
        cdfepsilon=np.cumsum(histepsilon)
        cdfepsilon=cdfepsilon / cdfepsilon[-1]
        valuesepsilon = np.random.rand(1)
        value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
        epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

        histsigma,binssigma = np.histogram(sigmamat)
        bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
        cdfsigma=np.cumsum(histsigma)
        cdfsigma=cdfsigma / cdfsigma[-1]
        valuessigma = np.random.rand(1)
        value_binssigma = np.searchsorted(cdfsigma,valuessigma)
        sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)

        histseasonstart,binsseasonstart = np.histogram(seasonstartmat)
        bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
        cdfseasonstart=np.cumsum(histseasonstart)
        cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
        valuesseasonstart = np.random.rand(1)
        value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
        seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

        histseasonend,binsseasonend = np.histogram(seasonendmat)
        bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
        cdfseasonend=np.cumsum(histseasonend)
        cdfseasonend=cdfseasonend / cdfseasonend[-1]
        valuesseasonend = np.random.rand(1)
        value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
        seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)

        mu1=((1/16)/(7/9))-(1/16)
        mu2=1/16
        mu=1/(365*67) #natural birth and death rate
        E=random.choice(Emat) #intial number of infected individuals
        S=N-E #number of susceptibles
        I=1 #number of exposed
        R=0 #number of dead
        #    inputSEIR = [S, E, I, R]

        #    SEIR = []
        #    SEIR.append([t, S, E, I, R]) #array of SEIR values

        ifb=E #infection from bats
        ifh=0 #infection from human to human


            #main loop
        while t < T:
                    
            if t%365<seasonstart and t%365>seasonend:
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
                ifh = ifh + 1
                a = a + 1
                        
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
                ifb = ifb + 1
                
            else:
                R = R - 1
                
                
    ifbmat.append(ifb)
    ifhmat.append(ifh)
    ifmat.append(ifb+ifh)
    deadmat.append(R)                
                    
    #        SEIR.append([t, S, E, I, R]) #adds data to SIR matrix

                
    #    time = [row[0] for row in SEIR] #times to be plotted
    #    susceptible = [row[1] for row in SEIR] #susceptible individuals to be plotted
    #    exposed = [row[2] for row in SEIR]
    #    infected = [row[3] for row in SEIR] #infectious individuals to be plotted
    #    dead = [row[4] for row in SEIR] #dead individuals to be plotted

    #    plt.step(time, exposed, 'b', label='Exposed')
    #    plt.step(time, infected, 'y', label='Infected')
    #    plt.step(time, dead, 'r', label='Dead')
            
    #    plt.xlabel('Time in Days (0 = Jan 1st)')
    #    plt.ylabel('Population')
      

        
        
print('Average number of total infections is',np.mean(ifmat))
print('Average number of total infections from bats is',np.mean(ifbmat))
print('Average number of total infections from humans is',np.mean(ifhmat))
print('Standard deviation of number of total infections is',np.std(ifmat))

#colors = ['blue', 'yellow', 'red']
#lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors]
#labels = ['Exposed', 'Infectious', 'Dead']
#plt.legend(lines, labels)

#plt.show()



