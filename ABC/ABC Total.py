import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import seaborn as sns; sns.set()
import scipy.stats as stats
from scipy.stats import gamma

#function that creates the real days passed and real cumulative infections arrays
def days(df):
    global idays
    global cases
    idays=[] #real days array
    cases=[] #real cumulative number of cases array
    for i in range(len(df.index)):
        idays.append(df.loc[i, 'Day'])
        cases.append(df.loc[i, 'Cumulative number of cases'])
        
#Function that computes the error between real and simulated outbreaks
#Also alters the discrete time (in days) array for both real and simulated data sets so they're the same length and can be compared
#Finally alters the cumulative number of infections in the real and simulated case so that they're comparable
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
    caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))  #computes error     
       



#Function that Gillespie algorithm
def SEIR(N, beta, epsilon, sigma, mu1, mu2, E, seasonstart, seasonend, idays):
    global cumifmat
    global itaumat
    global cumdmat
    global dtaumat
    T=10000 #maximum elapsed time
    itau=idays[0] #starts simulated day count at the same time as real days
    t=idays[0] #start time (day first infection was seen)
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
                    
        N = S + E + I + R #total population
        rate1 = (beta * I * S)/N #human to human transmission
        rate2 = sigma * E #move from exposed to infected
        rate3 = mu1 * I #recovery
        rate4 = mu * N + mu2 * I #birth rate
        rate5 = mu * S #natural death from susceptible class
        rate6 = mu * E #natural death from exposed class
        rate7 = (mu+mu2) * I #natural death from infectious class
        rate8 = _epsilon * S #bat to human transmission
        rate9 = mu * R #natural death from recovered class
        ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9]
        ratetotal = sum(ratemat)
                    
        dt = -math.log(random.uniform(0,1))/ratetotal #timestep
        t = t + dt
                    
        r=random.uniform(0,1) #random number to choose what happens next   
                   
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
            
particles=200 #number of parameters sampled at each iteration
iterations=15

data=[] #data storage array
df1=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2004')
df2=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Tangail2005')
df3=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Thakurgaon2007')
df4=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Manikgonj2008')    
df5=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajbari2008')
df6=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2010')
df7=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2011')   
df8=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Joypurhat2012')
df9=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rajshiahi2012')
df10=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2014')
df11=pd.read_excel (r'C:\Users\alexr\Desktop\URSS Project\Datasets.xlsx', sheet_name='Rangpur2014')


#this is where parameters are picked, [a,b] is a uniform distribution from a to b inclusive.
betamatABC=[0,100]
epsilonmatABC=[0,0.01]
sigmamatABC=[0,0.25]
seasonendmatABC=[250, 364]
seasonstartmatABC=[0,150]
N=1500
mu2=1/16
E1matABC=list(np.arange(37)) #as 36 infections in Faridpur 2004
E2matABC=list(np.arange(12)) #as 11 infections in...
E3matABC=list(np.arange(8))
E4matABC=list(np.arange(5))
E5matABC=list(np.arange(7))
E6matABC=list(np.arange(18))
E7matABC=list(np.arange(8))
E8matABC=list(np.arange(13))
E9matABC=list(np.arange(4))
E10matABC=list(np.arange(6))
E11matABC=list(np.arange(5))

msemat_=[119, 28, 11, 2, 8, 91, 22, 15, 5, 7, 5] #matrix for initial accepted errors

for g in range(iterations):
        
    E1mat=[]
    msemat1=[]
    E2mat=[]
    msemat2=[]
    E3mat=[]
    msemat3=[]
    E4mat=[]
    msemat4=[]
    E5mat=[]
    msemat5=[]
    E6mat=[]
    msemat6=[]
    E7mat=[]
    msemat7=[]
    E8mat=[]
    msemat8=[]
    E9mat=[]
    msemat9=[]
    E10mat=[]
    msemat10=[]
    E11mat=[]
    msemat11=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
    seasonstartmat=[]
    seasonendmat=[]
    
    d=0
    
    for b in range(particles):

        i=0
        
        while i==0:
         
            #samples parameters from previous raw distributions
            histbeta,binsbeta = np.histogram(betamatABC, bins=math.ceil(math.sqrt(len(betamatABC))))
            bin_midpointsbeta = binsbeta[:-1]+np.diff(binsbeta)/2
            cdfbeta=np.cumsum(histbeta)
            cdfbeta=cdfbeta / cdfbeta[-1]
            valuesbeta = np.random.rand(1)
            value_binsbeta = np.searchsorted(cdfbeta,valuesbeta)
            beta = bin_midpointsbeta[value_binsbeta] + random.uniform((binsbeta[0]-binsbeta[1])/2,(binsbeta[1]-binsbeta[0])/2)
            print(beta)
            histepsilon,binsepsilon = np.histogram(epsilonmatABC, bins=math.ceil(math.sqrt(len(epsilonmatABC))))
            bin_midpointsepsilon = binsepsilon[:-1]+np.diff(binsepsilon)/2
            cdfepsilon=np.cumsum(histepsilon)
            cdfepsilon=cdfepsilon / cdfepsilon[-1]
            valuesepsilon = np.random.rand(1)
            value_binsepsilon = np.searchsorted(cdfepsilon,valuesepsilon)
            epsilon = bin_midpointsepsilon[value_binsepsilon] + random.uniform((binsepsilon[0]-binsepsilon[1])/2,(binsepsilon[1]-binsepsilon[0])/2)
        
            histsigma,binssigma = np.histogram(sigmamatABC, bins=math.ceil(math.sqrt(len(sigmamatABC))))
            bin_midpointssigma = binssigma[:-1]+np.diff(binssigma)/2
            cdfsigma=np.cumsum(histsigma)
            cdfsigma=cdfsigma / cdfsigma[-1]
            valuessigma = np.random.rand(1)
            value_binssigma = np.searchsorted(cdfsigma,valuessigma)
            sigma = bin_midpointssigma[value_binssigma] + random.uniform((binssigma[0]-binssigma[1])/2,(binssigma[1]-binssigma[0])/2)
                    
            histseasonstart,binsseasonstart = np.histogram(seasonstartmatABC, bins=math.ceil(math.sqrt(len(seasonstartmatABC))))
            bin_midpointsseasonstart = binsseasonstart[:-1]+np.diff(binsseasonstart)/2
            cdfseasonstart=np.cumsum(histseasonstart)
            cdfseasonstart=cdfseasonstart / cdfseasonstart[-1]
            valuesseasonstart = np.random.rand(1)
            value_binsseasonstart = np.searchsorted(cdfseasonstart,valuesseasonstart)
            seasonstart = bin_midpointsseasonstart[value_binsseasonstart] + random.uniform((binsseasonstart[0]-binsseasonstart[1])/2,(binsseasonstart[1]-binsseasonstart[0])/2)

            histseasonend,binsseasonend = np.histogram(seasonendmatABC, bins=math.ceil(math.sqrt(len(seasonendmatABC))))
            bin_midpointsseasonend = binsseasonend[:-1]+np.diff(binsseasonend)/2
            cdfseasonend=np.cumsum(histseasonend)
            cdfseasonend=cdfseasonend / cdfseasonend[-1]
            valuesseasonend = np.random.rand(1)
            value_binsseasonend = np.searchsorted(cdfseasonend,valuesseasonend)
            seasonend = bin_midpointsseasonend[value_binsseasonend] + random.uniform((binsseasonend[0]-binsseasonend[1])/2,(binsseasonend[1]-binsseasonend[0])/2)     
 
 
            #runs simulation against Fairdpur 2004 outbreak once for this set of parameters and computes error
            days(df1)
            mu1=((1/16)/(7/9))-(1/16)   
            E1=random.choice(E1matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E1, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse1=caseserror
            print(mse1)
            
            days(df2)
            mu1=((1/16)/(10/11))-(1/16)    
            E2=random.choice(E2matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E2, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse2=caseserror
            
            days(df3)
            mu1=((1/16)/(3/7))-(1/16)    
            E3=random.choice(E3matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E3, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse3=caseserror
            
            days(df4)
            mu1=((1/16)/(3/4))-(1/16)    
            E4=random.choice(E4matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E4, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse4=caseserror
            
            days(df5)
            mu1=((1/16)/(3/4))-(1/16)    
            E5=random.choice(E5matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E5, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse5=caseserror
            
            days(df6)
            mu1=((1/16)/(3/4))-(1/16)    
            E6=random.choice(E6matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E6, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse6=caseserror
            
            days(df7)
            mu1=((1/16)/(3/4))-(1/16)   
            E7=random.choice(E7matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E7, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse7=caseserror
            
            days(df8)
            mu1=((1/16)/(10/12))-(1/16)    
            E8=random.choice(E8matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E8, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse8=caseserror
            
            days(df9)
            mu1=((1/16)/(3/4))-(1/16)    
            E9=random.choice(E9matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E9, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse9=caseserror
            
            days(df10)
            mu1=((1/16)/(4/5))-(1/16)    
            E10=random.choice(E10matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E10, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse10=caseserror
            
            days(df11)
            mu1=((1/16)/(3/4))-(1/16)    
            E11=random.choice(E11matABC)            
            SEIR(N, beta, epsilon, sigma, mu1, mu2, E11, seasonstart, seasonend, idays)
            error(itaumat, cumifmat, idays, cases)
            mse11=caseserror
            
            if mse1<msemat_[0] and mse2<msemat_[1] and mse3<msemat_[2] and mse4<msemat_[3] and mse5<msemat_[4] and mse6<msemat_[5] and mse7<msemat_[6] and mse8<msemat_[7] and mse9<msemat_[8] and mse10<msemat_[9] and mse11<msemat_[10]:
                i=i+1
                betamat.append(list(beta)[0])
                epsilonmat.append(list(epsilon)[0])
                sigmamat.append(list(sigma)[0])
                seasonstartmat.append(list(seasonstart)[0])
                seasonendmat.append(list(seasonend)[0])
                E1mat.append(E1)
                msemat1.append(mse1)
                E2mat.append(E2)
                msemat2.append(mse2)
                E3mat.append(E3)
                msemat3.append(mse3)
                E4mat.append(E4)
                msemat4.append(mse4)
                E5mat.append(E5)
                msemat5.append(mse5)
                E6mat.append(E6)
                msemat6.append(mse6)
                E7mat.append(E7)
                msemat7.append(mse7)
                E8mat.append(E8)
                msemat8.append(mse8)
                E9mat.append(E9)
                msemat9.append(mse9)
                E10mat.append(E10)
                msemat10.append(mse10)
                E11mat.append(E11)
                msemat11.append(mse11)
                msemat.append(mse)
                d=d+1
                print(d)
                
            else:
                i=0
             
        betamatABC=betamat
        epsilonmatABC=epsilonmat
        sigmamatABC=sigmamat
        seasonstartmatABC=seasonstartmat
        seasonendmatABC=seasonendmat
        E1matABC=E1mat
        E2matABC=E2mat
        E3matABC=E3mat
        E4matABC=E4mat
        E5matABC=E5mat
        E6matABC=E6mat
        E7matABC=E7mat
        E8matABC=E8mat
        E9matABC=E9mat
        E10matABC=E10mat
        E11matABC=E11mat
        msemat_=[np.median(msemat1), np.median(msemat2), np.median(msemat3), np.median(msemat4), np.median(msemat5), np.median(msemat6), np.median(msemat7), np.median(msemat8), np.median(msemat9), np.median(msemat10), np.median(msemat11)]

        data.append(betamat)
        data.append(epsilonmat)
        data.append(sigmamat)
        data.append(seasonstartmat)
        data.append(seasonendmat)
        data.append(E1mat)
        data.append(msemat1)
        data.append(E2mat)
        data.append(msemat2)
        data.append(E3mat)
        data.append(msemat3)
        data.append(E4mat)
        data.append(msemat4)
        data.append(E5mat)
        data.append(msemat5)
        data.append(E6mat)
        data.append(msemat6)
        data.append(E7mat)
        data.append(msemat7)
        data.append(E8mat)
        data.append(msemat8)
        data.append(E9mat)
        data.append(msemat9)
        data.append(E10mat)
        data.append(msemat10)
        data.append(E11mat)
        data.append(msemat11)        
                
        print(betamat)
        print(epsilonmat)
        print(sigmamat)
        print(seasonstartmat)
        print(seasonendmat)
        print(msemat1)
        print(E1mat)
        print(msemat2)
        print(E2mat)
        print(msemat3)
        print(E3mat)
        print(msemat4)
        print(E4mat)
        print(msemat5)
        print(E5mat)
        print(msemat6)
        print(E6mat)
        print(msemat7)
        print(E7mat)
        print(msemat8)
        print(E8mat)
        print(msemat9)
        print(E9mat)
        print(msemat10)
        print(E10mat)
        print(msemat11)
        print(E11mat)
                
data=np.column_stack(data)
np.savetxt('TotalABCdata.dat', data)