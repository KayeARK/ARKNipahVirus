import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.lines import Line2D
from statistics import mode

#loads file with each iteration of ABC data
data = np.loadtxt('C:\\Users\\Alex\\Desktop\\URSS Project\\GitHub\\Parameter Results\\Individual Outbreak Data Sets\\Faridpur2004ABCdata.dat')


#puts the "best" (last) batch of ABC data into arrays
betamat=data[:,98]
epsilonmat=data[:,99]
sigmamat=data[:,100]
seasonstartmat=data[:,101]
seasonendmat=data[:,102]
msemat=data[:,103]
Emat=data[:,104]

p=-0.04
pmat=[]
avgiftmattotal=[]
modeiftmattotal=[]

for l in range(26): #runs algorithm for increasing vaccine efficacy
    
    p=p+0.04
    pmat.append(p)
    print(p)

    gammamat=[]
    avgiftmat=[]
    modeiftmat=[]

    gamma = -4

    for f in range(26): #runs algorithm for increasing number of people vaccinated per day 
        
        ifbmat=[]
        ifhmat=[]
        ifmat=[]
        gamma=gamma+4
        gammamat.append(gamma)
        print(gamma)

        for i in range(200): #runs simulation 200 times for each value of gamma and p (might be worth increasing to 500/1000)
            
            N=1500 #population of village
            T=500 #elapsed time

            t=50        
             
            #the following generates distributions for each parameter from which parameter values are picked at random in the control simulation
            
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



            Vnumber=1500 #the total number of vaccines available to use
            mu1=((1/16)/(7/9))-(1/16) #recovery rate
            mu2=1/16 #death rate due to disease
            mu=1/(365*67) #natural birth and death rate
            E=random.choice(Emat) #intial number of infected individuals
            S=N-E #number of susceptibles
            I=1 #number of exposed
            R=0 #number of dead
            VF=0 #intital number of failed vaccinations
            VS=0 #intital number of sucessful vaccinations
            inputSEIVR = [S, E, I, VF, VS, R]

            SEIVR = []
            SEIVR.append([t, S, E, I, VF, VS, R]) #array of SEIR values

            ifb=0 #infection from bats
            ifh=0 #infection from human to human


            #main loop
            while t < T:
                        
                if t%365<seasonstart and t%365>seasonend:
                    _epsilon=0
                               
                else:
                    _epsilon=epsilon
                    
                if Vnumber<1:
                    _gamma=0
                    
                else:
                    _gamma=gamma
                
                if I==0 and E==0 and _epsilon==0:
                    break
                
                if S+E==0:
                    vfr=0
                    vsr=0
                    
                else:
                    vfr=((1 - p) * _gamma * S)/(S + E)
                    vsr=(p * _gamma * S)/(S + E)

                                         
                N = S + E + I + VF + VS + R
                rate1 = (beta * I * S)/N #human to human transmission
                rate2 = sigma * E #move from exposed to infected
                rate3 = (mu + mu2) * I #disease induced death
                rate4 = mu1 * I #recovery from the disease and move back to susceptible
                rate5 = mu * N + mu2*I #birth rate
                rate6 = mu * S #natural death from susceptible class
                rate7 = mu * E #natural death from exposed class
                rate8 = mu * R #natural death from infectious class
                rate9 = _epsilon * S #transmission from bats
                rate10 = mu * VF
                rate11 = mu * VS
                rate12= (beta * I * VF)/N
                rate13= _epsilon * VF
                rate14= vfr
                rate15= vsr
                    
                ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, rate10, rate11, rate12, rate13, rate14, rate15]
                ratetotal = sum(ratemat)
                        
                dt = -math.log(random.uniform(0,1))/ratetotal #timestep
                t = t + dt
                        
                r=random.uniform(0,1)    
                       
                if r < sum(ratemat [:1])/ratetotal: #move to infected class
                    S = S - 1
                    E = E + 1
                    ifh = ifh + 1
                            
                elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
                    E = E - 1
                    I = I + 1
                            
                elif sum(ratemat [:2])/ratetotal < r and r < sum(ratemat [:3])/ratetotal:
                    I = I - 1
                            
                elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
                    I = I - 1
                    R = R + 1
                            
                elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
                    S = S + 1
                            
                elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
                    S = S - 1
                            
                elif sum(ratemat [:6])/ratetotal < r and r < sum(ratemat [:7])/ratetotal:
                    E = E - 1
                            
                elif sum(ratemat [:7])/ratetotal < r and r < sum(ratemat [:8])/ratetotal:
                    R = R - 1              

                elif sum(ratemat [:8])/ratetotal < r and r < sum(ratemat [:9])/ratetotal:
                    S = S - 1
                    E = E + 1
                    ifb = ifb + 1
                    
                elif sum(ratemat [:9])/ratetotal < r and r < sum(ratemat [:10])/ratetotal:
                    VF = VF - 1
                    
                elif sum(ratemat [:10])/ratetotal < r and r < sum(ratemat [:11])/ratetotal:
                    VS = VS - 1
                    
                elif sum(ratemat [:11])/ratetotal < r and r < sum(ratemat [:12])/ratetotal:
                    VF = VF - 1
                    E = E + 1
                    ifh = ifh + 1
                    
                elif sum(ratemat [:12])/ratetotal < r and r < sum(ratemat [:13])/ratetotal:
                    VF = VF - 1
                    E = E + 1
                    ifb = ifb + 1
                    
                elif sum(ratemat [:13])/ratetotal < r and r < sum(ratemat [:14])/ratetotal:
                    S = S - 1
                    VF = VF + 1
                    Vnumber=Vnumber-1
                    
                else:
                    S = S - 1
                    VS = VS + 1
                    Vnumber=Vnumber-1

                            
                SEIVR.append([t, S, E, I, VF, VS, R]) #adds data to SIR matrix

            ifbmat.append(ifb)
            ifhmat.append(ifh)
            ifmat.append(ifb+ifh)

        modeiftmat.append(max([i for i in set(ifmat) if ifmat.count(i) == max(map(ifmat.count, ifmat))]))
        avgiftmat.append(np.mean(ifmat))

    avgiftmattotal.append(avgiftmat)
    modeiftmattotal.append(modeiftmat)


#prints data, might be best to save it wherever is fit
print("Mean Infection Data:", avgiftmattotal)
print("Mode Infection Data:", modeiftmattotal)
print("p Values:", pmat)
print("gamma Values:", gammamat)