import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import seaborn as sns; sns.set()
import scipy.stats as stats
from scipy.stats import gamma

def days(df):
    global idays
    global cases
    idays=[]
    cases=[]
    for i in range(len(df.index)):
        idays.append(df.loc[i, 'Day'])
        cases.append(df.loc[i, 'Cumulative number of cases'])
        
        
def error(itaumat, cumifmat, idays, cases):
    global caseserror
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
       




def SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays):
    global cumifmat
    global itaumat
    global cumdmat
    global dtaumat
    T=10000 #elapsed time
    itau=idays[0]
    t=idays[0]
    I=cases[0] #intial number of infected individuals
    S=N-I #number of susceptibles
    R=0 #number of dead
    ift=I
    mu=1/(365*67) #natural birth and death rate
    itaumat=[idays[0]]
    cumifmat=[I]
    
    while t < T:
        
        if I > 100:
            break
                 
        if t%365<seasonstart and t%365>seasonend:
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
        rate5 = mu * N + mu2*I #birth rate
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


#df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2004')
'''
print("Faridpur 2004 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,200)
        mu1=((1/16)/(10/11))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)

'''













df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Tangail2005')

print("Tangail2005 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,200)
        mu1=((1/16)/(10/11))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)









df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Thakurgaon2007')

print("Thakurgaon2007 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/7))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    





df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Manikgonj2008')

print("Manikgonj2008 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75% as unknown
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    
    
    
    
    
    
df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajbari2008') 

print("Rajbari2008 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    






df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2010')

print("Faridpur2010 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)









df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2011')

print("Faridpur2011 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(4/5))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    






df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Lalmonirhat2011')

print("Lalmonirhat2011 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(21/22))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    





df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2011')

print("Rangpur2011 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    
    
    
    
df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Joypurhat2012')   

print("Joypurhat2012 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(10/12))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)





df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajshiahi2012')

print("Rajshiahi2012 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)








df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2014')

print("Faridpur2014 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(4/5))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)









df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2014')

print("Rangpur2014 Results")
betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
Emat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=10000

    while mse>100:
        
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(250,365)
        seasonend=random.uniform(0,100)
        mu1=((1/16)/(3/4))-(1/16) #assumed death rate of 75%
        mu2=1/16        
        
        days(df)
        E=random.randint(0,math.floor(cases[-1]))
        N=df.loc[0,'Population']
        SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
        error(itaumat, cumifmat, idays, cases)
        c1=caseserror

        mse=c1
    
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
    seasonstartmat.append(seasonstart)
    seasonendmat.append(seasonend)
    Emat.append(E)
    msemat.append(mse)

print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
print('seasonstartmat1','=',seasonstartmat)
print('seasonendmat1','=',seasonendmat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)


betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
seasonstartmatABC=seasonstartmat
seasonendmatABC=seasonendmat
msematABC=msemat
EmatABC=Emat

for c in range(14):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=10000

        while mse>mse_:
     
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

            days(df)
            E=random.choice(EmatABC)
            N=df.loc[0,'Population']
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            c1=caseserror

            mse=c1

        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
        seasonstartmat.append(seasonstart)
        seasonendmat.append(seasonend)
        Emat.append(E)
        msemat.append(mse)
 
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
    seasonstartmatABC=seasonstartmat
    seasonendmatABC=seasonendmat
    msematABC=msemat
    EmatABC=Emat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
    print('seasonstartmat',c+2,'=',seasonstartmat)
    print('seasonendmat',c+2,'=',seasonendmat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)


#FIN