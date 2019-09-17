import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import seaborn as sns; sns.set()
import scipy.stats as stats
from scipy.stats import gamma
 

def SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend):
    global cumifmat
    global itaumat
    global cumdmat
    global dtaumat
    T=10000 #elapsed time
    itau=idays[0]
    I=cases[0] #intial number of infected individuals
    S=N-I #number of susceptibles
    R=0 #number of dead
    ift=I #total number of infected counter
    mu=1/(365*67) #natural birth and death rate, 67 years
    itaumat=[idays[0]] #simulated days array
    cumifmat=[I] #simulated cumulative infection array

    
    while t < T:
        
        if I > 100: #stops the simulation if number of infected is greater than 100, as this is way too high
            break
                 
        if t%365<seasonstart and t%365>seasonend: #makes epsilon seasonal
            _epsilon=0
                           
        else:
            _epsilon=epsilon
            
        if I==0 and E==0 and _epsilon==0: #only ends the algorithm when there is no potential for a new infection (including not date palm season)
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
        rate9 = mu * R #natural death from recovered class
        ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        ratetotal = sum(ratemat)
                   
        dt = -math.log(random.uniform(0,1))/ratetotal #timestep
        t = t + dt
                    
        r=random.uniform(0,1)    
                   
        if r < sum(ratemat [:1])/ratetotal:
            S = S - 1
            E = E + 1
                      
        elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
            E = E - 1
            I = I + 1
            ift = ift + 1
            itau=math.floor(t)
            itaumat.append(itau)
            cumifmat.append(ift)
                        
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


#df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2004')
'''
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(7/9))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>119: #erorr acceptance is 119
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1]) #selects initial number of exposed from uniform distribution between 0 and the total number of infections inclusive
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)


        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#31:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Faridpur2004ABCdata.dat', data)
'''









#df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Tangail2005')
'''
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(10/11))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>28:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#5:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Tangail2005ABCdata.dat', data)
'''






#df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Thakurgaon2007')
'''
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/7))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>11:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#4:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Thakurgaon2007ABCdata.dat', data)
'''










df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Manikgonj2008')

data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>2:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#1:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Manikgonj2008ABCdata.dat', data)








df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajbari2008')

data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>8:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#2:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Rajbari2008ABCdata.dat', data)








df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2010')

data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>91:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#21:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Faridpur2010ABCdata.dat', data)














df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2011')
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>22:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#5:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Rangpur2011ABCdata.dat', data)










df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Joypurhat2012')
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(10/12))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>15:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#5:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Joypurhat2012ABCdata.dat', data)









df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajshiahi2012')
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>5:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#2:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Rajshiahi2012ABCdata.dat', data)











df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2014')
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(4/5))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>7:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#3:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Faridpur2014ABCdata.dat', data)








df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2014')
data=[]
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

mu1=((1/16)/(3/4))-(1/16)
mu2=1/16

iterations=200

for b in range(iterations):

    mse=10000

    while mse>5:
        
        idays=[]
        cases=[]


        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])

            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=1500
        beta=random.uniform(0,1)
        epsilon=random.uniform(0,0.01)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,364)
        seasonend=random.uniform(t,150)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)


        i=0

        while i < len(itaumat)-1:
            if itaumat[i]==itaumat[i+1]:
                del cumifmat[i]
                del itaumat[i]                
                                          
            elif itaumat[i]+1!=itaumat[i+1]:
                itaumat.insert(i+1, itaumat[i]+1)
                cumifmat.insert(i+1, cumifmat[i])

            else:
                i = i + 1                             
                                              
        while len(itaumat)<len(idays):
            itaumat.append(itaumat[-1]+1)
            cumifmat.append(cumifmat[-1])        
       
        while len(itaumat)>len(idays):
            idays.append(idays[-1]+1)
            cases.append(cases[-1])        

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        mse=caseserror        
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)
    print(b)
    
data.append(betamat)
data.append(epsilonmat)
data.append(sigmamat)
data.append(seasonstartmat)
data.append(seasonendmat)
data.append(msemat)
data.append(Emat)

print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
print('seasonstartmat1','=',list(seasonstartmat))
print('seasonendmat1','=',list(seasonendmat))
print('msemat1','=',list(msemat))
print('Emat1','=',list(Emat))


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat
mse_=np.median(msematABC)
c=1
for f in range(14):#2:
    
    
    iteration=[]
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            cases=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                                  
            E=random.choice(EmatABC)
            t=idays[0]
            N=1500
            
            histbeta,binsbeta = np.histogram(betamatABC)
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)

            histepsilon,binsepsilon = np.histogram(epsilonmatABC)
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)

            histsigma,binssigma = np.histogram(sigmamatABC)
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
            
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC)
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC)
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)
         
            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend)

            i=0

            while i < len(itaumat)-1:
                if itaumat[i]==itaumat[i+1]:
                    del cumifmat[i]
                    del itaumat[i]                
                                              
                elif itaumat[i]+1!=itaumat[i+1]:
                    itaumat.insert(i+1, itaumat[i]+1)
                    cumifmat.insert(i+1, cumifmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(itaumat)<len(idays):
                itaumat.append(itaumat[-1]+1)
                cumifmat.append(cumifmat[-1])        
           
            while len(itaumat)>len(idays):
                idays.append(idays[-1]+1)
                cases.append(cases[-1])

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            mse=caseserror           
       
        betamat.append(list(beta)[0])
        epsilonmat.append(list(epsilon)[0])
        sigmamat.append(list(sigma)[0])
        seasonstartmat.append(list(seasonstart)[0])
        seasonendmat.append(list(seasonend)[0])
        Emat.append(E)
        msemat.append(mse)
   
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+1,'=',list(betamat))
    print('epsilonmat',c+1,'=',list(epsilonmat))
    print('sigmamat',c+1,'=',list(sigmamat))
    print('seasonstartmat',c+1,'=',list(seasonstartmat))
    print('seasonendmat',c+1,'=',list(seasonendmat))
    print('msemat',c+1,'=',list(msemat))
    print('Emat',c+1,'=',list(Emat))
    
    data.append(betamat)
    data.append(epsilonmat)
    data.append(sigmamat)
    data.append(seasonstartmat)
    data.append(seasonendmat)
    data.append(msemat)
    data.append(Emat)
    
    c=c+1
    iteration.append(c)
    
data=np.column_stack(data)
np.savetxt('Rangpur2014ABCdata.dat', data)

'''
data = np.loadtxt('Faridpur2004ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(c):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])
'''

