import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pylab


'''
        E=random.randint(0,cases[-1
        t=idays[0]
        N=df.loc[0,'Population']
        beta=random.uniform(0,5)
        epsilon=random.uniform(0,0.001)
        sigma=random.uniform(0,1/4)
        seasonstart=random.uniform(325,365)
        seasonend=random.uniform(40,100)
        mu1=((1/16)/(7/9))-(1/16)
        mu2=1/16
'''






seventeenmat=17*np.ones((200,),dtype=int)
sixteenmat=16*np.ones((200,),dtype=int)
fifteenmat=15*np.ones((200,),dtype=int)
fourteenmat=14*np.ones((200,),dtype=int)
thirteenmat=13*np.ones((200,),dtype=int)
twelvemat=12*np.ones((200,),dtype=int)
elevenmat=11*np.ones((200,),dtype=int)
tenmat=10*np.ones((200,),dtype=int)
ninemat=9*np.ones((200,),dtype=int)
eightmat=8*np.ones((200,),dtype=int)
sevenmat=7*np.ones((200,),dtype=int)
sixmat=6*np.ones((200,),dtype=int)
fivemat=5*np.ones((200,),dtype=int)
fourmat=4*np.ones((200,),dtype=int)
threemat=3*np.ones((200,),dtype=int)
twomat=2*np.ones((200,),dtype=int)
onemat=1*np.ones((200,),dtype=int)


fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC')

plt.subplot(2, 3, 1)
plt.scatter(betamat1,seventeenmat,s=4)
plt.scatter(betamat2,sixteenmat,s=4)
plt.scatter(betamat3,fifteenmat,s=4)
plt.scatter(betamat4,fourteenmat,s=4)
plt.scatter(betamat5,thirteenmat,s=4)
plt.scatter(betamat6,twelvemat,s=4)
plt.scatter(betamat7,elevenmat,s=4)
plt.scatter(betamat8,tenmat,s=4)
plt.scatter(betamat9,ninemat,s=4)
plt.scatter(betamat10,eightmat,s=4)
plt.scatter(betamat11,sevenmat,s=4)
plt.scatter(betamat12,sixmat,s=4)
plt.scatter(betamat13,fivemat,s=4)
plt.scatter(betamat14,fourmat,s=4)
plt.scatter(betamat15,threemat,s=4)
#plt.scatter(betamat16,twomat,s=4)
#plt.scatter(betamat17,onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat1,seventeenmat,s=4)
plt.scatter(epsilonmat2,sixteenmat,s=4)
plt.scatter(epsilonmat3,fifteenmat,s=4)
plt.scatter(epsilonmat4,fourteenmat,s=4)
plt.scatter(epsilonmat5,thirteenmat,s=4)
plt.scatter(epsilonmat6,twelvemat,s=4)
plt.scatter(epsilonmat7,elevenmat,s=4)
plt.scatter(epsilonmat8,tenmat,s=4)
plt.scatter(epsilonmat9,ninemat,s=4)
plt.scatter(epsilonmat10,eightmat,s=4)
plt.scatter(epsilonmat11,sevenmat,s=4)
plt.scatter(epsilonmat12,sixmat,s=4)
plt.scatter(epsilonmat13,fivemat,s=4)
plt.scatter(epsilonmat14,fourmat,s=4)
plt.scatter(epsilonmat15,threemat,s=4)
#plt.scatter(epsilonmat16,twomat,s=4)
#plt.scatter(epsilonmat17,onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat1,seventeenmat,s=4)
plt.scatter(sigmamat2,sixteenmat,s=4)
plt.scatter(sigmamat3,fifteenmat,s=4)
plt.scatter(sigmamat4,fourteenmat,s=4)
plt.scatter(sigmamat5,thirteenmat,s=4)
plt.scatter(sigmamat6,twelvemat,s=4)
plt.scatter(sigmamat7,elevenmat,s=4)
plt.scatter(sigmamat8,tenmat,s=4)
plt.scatter(sigmamat9,ninemat,s=4)
plt.scatter(sigmamat10,eightmat,s=4)
plt.scatter(sigmamat11,sevenmat,s=4)
plt.scatter(sigmamat12,sixmat,s=4)
plt.scatter(sigmamat13,fivemat,s=4)
plt.scatter(sigmamat14,fourmat,s=4)
plt.scatter(sigmamat15,threemat,s=4)
#plt.scatter(sigmamat16,twomat,s=4)
#plt.scatter(sigmamat17,onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat1,seventeenmat,s=4)
plt.scatter(Emat2,sixteenmat,s=4)
plt.scatter(Emat3,fifteenmat,s=4)
plt.scatter(Emat4,fourteenmat,s=4)
plt.scatter(Emat5,thirteenmat,s=4)
plt.scatter(Emat6,twelvemat,s=4)
plt.scatter(Emat7,elevenmat,s=4)
plt.scatter(Emat8,tenmat,s=4)
plt.scatter(Emat9,ninemat,s=4)
plt.scatter(Emat10,eightmat,s=4)
plt.scatter(Emat11,sevenmat,s=4)
plt.scatter(Emat12,sixmat,s=4)
plt.scatter(Emat13,fivemat,s=4)
plt.scatter(Emat14,fourmat,s=4)
#plt.scatter(Emat15,threemat,s=4)
#plt.scatter(Emat16,twomat,s=4)
#plt.scatter(Emat17,onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat1,seventeenmat,s=4)
plt.scatter(seasonstartmat2,sixteenmat,s=4)
plt.scatter(seasonstartmat3,fifteenmat,s=4)
plt.scatter(seasonstartmat4,fourteenmat,s=4)
plt.scatter(seasonstartmat5,thirteenmat,s=4)
plt.scatter(seasonstartmat6,twelvemat,s=4)
plt.scatter(seasonstartmat7,elevenmat,s=4)
plt.scatter(seasonstartmat8,tenmat,s=4)
plt.scatter(seasonstartmat9,ninemat,s=4)
plt.scatter(seasonstartmat10,eightmat,s=4)
plt.scatter(seasonstartmat11,sevenmat,s=4)
plt.scatter(seasonstartmat12,sixmat,s=4)
plt.scatter(seasonstartmat13,fivemat,s=4)
plt.scatter(seasonstartmat14,fourmat,s=4)
#plt.scatter(seasonstartmat15,threemat,s=4)
#plt.scatter(seasonstartmat16,twomat,s=4)
#plt.scatter(seasonstartmat17,onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat1,seventeenmat,s=4)
plt.scatter(seasonendmat2,sixteenmat,s=4)
plt.scatter(seasonendmat3,fifteenmat,s=4)
plt.scatter(seasonendmat4,fourteenmat,s=4)
plt.scatter(seasonendmat5,thirteenmat,s=4)
plt.scatter(seasonendmat6,twelvemat,s=4)
plt.scatter(seasonendmat7,elevenmat,s=4)
plt.scatter(seasonendmat8,tenmat,s=4)
plt.scatter(seasonendmat9,ninemat,s=4)
plt.scatter(seasonendmat10,eightmat,s=4)
plt.scatter(seasonendmat11,sevenmat,s=4)
plt.scatter(seasonendmat12,sixmat,s=4)
plt.scatter(seasonendmat13,fivemat,s=4)
plt.scatter(seasonendmat14,fourmat,s=4)
#plt.scatter(seasonendmat15,threemat,s=4)
#plt.scatter(seasonendmat16,twomat,s=4)
#plt.scatter(Emat17,onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Rangpur2011ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter')

plt.subplot(3, 2, 1)
plt.hist(betamat15, density=1)
plt.xlabel('\u03B2')
plt.hist([0,0.05],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat15, density=1)
plt.xlabel('\u03B5')
plt.hist([0.0002,0.0005],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat15, density=1)
plt.xlabel('Season Start Time')
plt.hist([325,365],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat15, density=1)
plt.xlabel('Season End Time')
plt.hist([90,100],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat15, density=1)
plt.xlabel('\u03C3')
plt.hist([0.04,0.125],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Rangpur2011PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()



'''
fig = plt.figure()
plt.suptitle('Plots of Each Parameter against all other Parameters')

plt.subplot(2, 3, 1)
plt.scatter(betamat14,epsilonmat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.ylim(min(epsilonmat14),max(epsilonmat14))
plt.xlabel('\u03B2')
plt.ylabel('\u03B5')

plt.subplot(2, 3, 2)
plt.scatter(betamat14,sigmamat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlabel('\u03B2')
plt.ylabel('\u03C3')

plt.subplot(2, 3, 3)
plt.scatter(betamat14,Emat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlabel('\u03B2')
plt.ylabel('E')

plt.subplot(2, 3, 4)
plt.scatter(epsilonmat14,sigmamat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlim(min(epsilonmat14),max(epsilonmat14))
plt.xlabel('\u03B5')
plt.ylabel('\u03C3')

plt.subplot(2, 3, 5)
plt.scatter(epsilonmat14,Emat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlim(min(epsilonmat14),max(epsilonmat14))
plt.xlabel('\u03B5')
plt.ylabel('E')

plt.subplot(2, 3, 6)
plt.scatter(sigmamat14,Emat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlabel('\u03C3')
plt.ylabel('E')


plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=1)
plt.savefig('PlotsofEachParameteragainstallotherParameters.pdf',bbox_inches='tight',transparent = True)
plt.show()
'''