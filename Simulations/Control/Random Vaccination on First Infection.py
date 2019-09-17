import math
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
from matplotlib.lines import Line2D
from statistics import mode

p=-0.01
pmat=[]
avgiftmattotal=[]
modeiftmattotal=[]

for l in range(101):
    
    p=p+0.01
    pmat.append(p)
    print(p)

    gammamat=[]
    avgiftmat=[]
    modeiftmat=[]

    gamma = -1

    for f in range(101):
        
        ifbmat=[]
        ifhmat=[]
        ifmat=[]
        gamma=gamma+1
        gammamat.append(gamma)
        print(gamma)

        for i in range(100):
            
            N=1500 #population of village
            T=500 #elapsed time

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
            VF=0
            VS=0
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
                
                if I==0 and E==0 and _epsilon==0:
                    break
                
                if S+E==0:
                    vfr=0
                    
                else:
                    vfr=((1 - p) * gamma * S)/(S + E)
                    
                if S+E==0:
                    vsr=0
                    
                else:
                    vsr=(p * gamma * S)/(S + E)
                                         
                N = S + E + I + VF + VS
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
                    ifb = ifh + 1
                    
                elif sum(ratemat [:12])/ratetotal < r and r < sum(ratemat [:13])/ratetotal:
                    VF = VF - 1
                    E = E + 1
                    ifb = ifb + 1
                    
                elif sum(ratemat [:13])/ratetotal < r and r < sum(ratemat [:14])/ratetotal:
                    S = S - 1
                    VF = VF + 1
                    
                else:
                    S = S - 1
                    VS = VS + 1                

                            
                SEIVR.append([t, S, E, I, VF, VS, R]) #adds data to SIR matrix

            ifbmat.append(ifb)
            ifhmat.append(ifh)
            ifmat.append(ifb+ifh)

        modeiftmat.append(max([i for i in set(ifmat) if ifmat.count(i) == max(map(ifmat.count, ifmat))]))
        avgiftmat.append(np.mean(ifmat))

    avgiftmattotal.append(avgiftmat)
    modeiftmattotal.append(modeiftmat)
    
print("Mean Infection Data:", avgiftmattotal)
print("Mode Infection Data:", modeiftmattotal)
print("p Values:", pmat)
print("gamma Values:", gammamat)

'''                
time = [row[0] for row in SEIVR] #times to be plotted
susceptible = [row[1] for row in SEIVR] #susceptible individuals to be plotted
exposed = [row[2] for row in SEIVR]
infected = [row[3] for row in SEIVR] #infectious individuals to be plotted
vaccinatedf = [row[4] for row in SEIVR]
vaccinateds= [row[5] for row in SEIVR]
recovered = [row[6] for row in SEIVR] #dead individuals to be plotted

plt.step(time, exposed, 'b', label='Exposed')
plt.step(time, infected, 'y', label='Infected')
plt.step(time, vaccinateds, 'g', label='Vacinated Succesfully')
plt.step(time, vaccinatedf, 'm', label='Vacinated Unsuccesfully')
plt.step(time, recovered, 'r', label='Recovered')
plt.legend()               
plt.xlabel('Time in Days (0 = Jan 1st)')
plt.ylabel('Population')
       
print('Average number of total infections is',np.mean(ifmat))
print('Average number of total infections from bats is',np.mean(ifbmat))
print('Average number of total infections from humans is',np.mean(ifhmat))
print('Standard deviation of number of total infections is',np.std(ifmat))
plt.show()

colors = ['blue', 'yellow', 'red', 'green', 'magenta']
lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors]
labels = ['Exposed', 'Infectious', 'Recovered', 'Vacinated Succesfully', 'Vacinated Unsuccesfully']
plt.legend(lines, labels)


def dSEIVR_dt(X, t):
    return [-epsilon*X[0]-(beta*X[0]*X[2]/N)+mu*(N-X[0])+mu2*X[2]-(gamma*X[0])/(X[0]+X[1]),epsilon*X[0]+(beta*X[0]*X[2]/N)+epsilon*X[3]+(beta*X[3]*X[2]/N)-sigma*X[1]-mu*X[1], sigma*X[1]-X[2]*(mu1+mu2+mu),(1-p)*gamma*X[0]/(X[0]+X[1])-mu*X[3]-epsilon*X[3]-(beta*X[3]*X[2]/N),(p*gamma*X[0])/(X[0]+X[1])-mu*X[4],mu1*X[2]-mu*X[5]]
tsolve=np.linspace(50, list(t)[0], 1000)

dSEIVR = odeint(dSEIVR_dt, inputSEIVR, tsolve)
plt.plot(tsolve, [row[1] for row in dSEIVR], '--', c='b')
plt.plot(tsolve, [row[2] for row in dSEIVR], '--', c='y')
plt.plot(tsolve, [row[3] for row in dSEIVR], '--', c='m')
plt.plot(tsolve, [row[4] for row in dSEIVR], '--', c='g')
plt.plot(tsolve, [row[5] for row in dSEIVR], '--', c='r')

plt.show()

print(gammamat)
print(avgiftmat)
print(modeiftmat)

gammamat=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
avgiftmat=[34.88, 35.39, 35.11333333333334, 35.145, 34.798, 34.22666666666667, 33.752857142857145, 33.2725, 32.74777777777778, 32.229, 31.774545454545454, 31.2975, 30.837692307692308, 30.31, 29.866666666666667, 29.421875, 28.92, 28.426111111111112, 27.999473684210525, 27.551, 27.163333333333334, 26.763636363636362, 26.307391304347828, 25.891666666666666, 25.4688, 25.036923076923078, 24.60037037037037, 24.196428571428573, 23.829310344827586, 23.453666666666667, 23.08451612903226, 22.73375, 22.37787878787879, 22.046176470588236, 21.728857142857144, 21.410555555555554, 21.102972972972974, 20.801052631578948, 20.492564102564103, 20.2225, 19.948536585365854, 19.68952380952381, 19.423488372093022, 19.176136363636363, 18.93777777777778, 18.70826086956522, 18.475106382978723, 18.244791666666668, 18.029591836734692, 17.8158, 17.613529411764706, 17.40653846153846, 17.19490566037736, 17.001296296296296, 16.814363636363638, 16.630535714285713, 16.457719298245614, 16.285862068965518, 16.113898305084746, 15.9415, 15.785573770491803, 15.62209677419355, 15.466825396825397, 15.3115625, 15.161692307692308, 15.013484848484849, 14.864029850746268, 14.72779411764706, 14.591304347826087, 14.45342857142857, 14.319154929577465, 14.187083333333334, 14.056301369863014, 13.929864864864864, 13.810133333333333, 13.688815789473685, 13.575584415584416, 13.462820512820512, 13.352151898734178, 13.24175, 13.136666666666667, 13.025487804878049, 12.920722891566266, 12.817380952380953, 12.717176470588235, 12.616860465116279, 12.519885057471264, 12.4225, 12.330449438202248, 12.239222222222223, 12.15065934065934, 12.06, 11.970322580645162, 11.881382978723405, 11.793894736842105, 11.706979166666667, 11.624329896907216, 11.543469387755103, 11.464949494949495, 11.388, 11.311089108910892]
modeiftmat=[37, 36, 36, 36, 36, 36, 28, 31, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 24, 24, 24, 24, 22, 22, 22, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 12, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]




plt.scatter(gammamat, avgiftmat, s=7, label='Average Total Number of Infections')
plt.scatter(gammamat, modeiftmat, s=7, label='Mode of Total Number of Infections')
plt.title("Average Number of Total Infections against Vaccinations per day")
plt.xlabel('\u03BB')
plt.ylabel('Average total number of infections')
plt.legend(loc='upper right')
#plt.savefig('AverageNumberofTotalInfectionsagainstVaccinationsperday.pdf',bbox_inches='tight',transparent = True)

plt.show()
'''