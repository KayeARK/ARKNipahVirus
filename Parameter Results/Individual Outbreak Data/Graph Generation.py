import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import pylab

data = np.loadtxt('Faridpur2004ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])


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
plt.suptitle('Parameter Values After Each Iteration of ABC - Faridpur 2004')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Faridpur2004ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Faridpur 2004')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,0.03],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.00065],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([40,120],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(Emat[14], density=1)
plt.xlabel('E')
plt.hist([0,2],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 6)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

#bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
#pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Faridpur2004PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()




data = np.loadtxt('Tangail2005ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Tangail 2005')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Tangail2005ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Tangail 2005')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Tangail2005PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()




data = np.loadtxt('Thakurgaon2007ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Thakurgaon 2007')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Thakurgaon2007ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Thakurgaon 007')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Thakurgaon2007PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Manikgonj2008ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Manikgonj 2008')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Manikgonj2008ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Manikgonj 2008')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Manikgonj2008PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Rajbari2008ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])



fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Rajbari 2008')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Rajbari2008ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Rajbari 2008')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Rajbari2008PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Faridpur2010ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Faridpur 2010')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Faridpur2010ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Faridpur 2010')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Faridpur2010PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Rangpur2011ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Rangpur 2011')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Rangpur2011ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Rangpur 2011')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Rangpur2011PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()




data = np.loadtxt('Joypurhat2012ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Joypurhat 2012')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Joypurhat2012ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Joypurhat 2012')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Joypurhat2012PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Rajshiahi2012ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Rajshiahi 2012')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Rajshiahi2012ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Rajshiahi 2012')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Rajshiahi2012PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()





data = np.loadtxt('Faridpur2014ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Faridpur 2014')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Faridpur2014ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Faridpur 2014')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Faridpur2014PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()







data = np.loadtxt('Rangpur2014ABCdata.dat')

betamat=[]
epsilonmat=[]
sigmamat=[]
seasonstartmat=[]
seasonendmat=[]
msemat=[]
Emat=[]

for b in range(15):
    betamat.append(data[:,7*b])
    epsilonmat.append(data[:,7*b+1])
    sigmamat.append(data[:,7*b+2])
    seasonstartmat.append(data[:,7*b+3])
    seasonendmat.append(data[:,7*b+4])
    msemat.append(data[:,7*b+5])
    Emat.append(data[:,7*b+6])





fig = plt.figure()
plt.suptitle('Parameter Values After Each Iteration of ABC - Rangpur 2014')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0],fifteenmat,s=4)
plt.scatter(betamat[1],fourteenmat,s=4)
plt.scatter(betamat[2],thirteenmat,s=4)
plt.scatter(betamat[3],twelvemat,s=4)
plt.scatter(betamat[4],elevenmat,s=4)
plt.scatter(betamat[5],tenmat,s=4)
plt.scatter(betamat[6],ninemat,s=4)
plt.scatter(betamat[7],eightmat,s=4)
plt.scatter(betamat[8],sevenmat,s=4)
plt.scatter(betamat[9],sixmat,s=4)
plt.scatter(betamat[10],fivemat,s=4)
plt.scatter(betamat[11],fourmat,s=4)
plt.scatter(betamat[12],threemat,s=4)
plt.scatter(betamat[13],twomat,s=4)
plt.scatter(betamat[14],onemat,s=4)
plt.xlabel('\u03B2')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 2)
plt.scatter(epsilonmat[0],fifteenmat,s=4)
plt.scatter(epsilonmat[1],fourteenmat,s=4)
plt.scatter(epsilonmat[2],thirteenmat,s=4)
plt.scatter(epsilonmat[3],twelvemat,s=4)
plt.scatter(epsilonmat[4],elevenmat,s=4)
plt.scatter(epsilonmat[5],tenmat,s=4)
plt.scatter(epsilonmat[6],ninemat,s=4)
plt.scatter(epsilonmat[7],eightmat,s=4)
plt.scatter(epsilonmat[8],sevenmat,s=4)
plt.scatter(epsilonmat[9],sixmat,s=4)
plt.scatter(epsilonmat[10],fivemat,s=4)
plt.scatter(epsilonmat[11],fourmat,s=4)
plt.scatter(epsilonmat[12],threemat,s=4)
plt.scatter(epsilonmat[13],twomat,s=4)
plt.scatter(epsilonmat[14],onemat,s=4)
plt.xlabel('\u03B5')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
#plt.xlim(0,0.0005)
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 3)
plt.scatter(sigmamat[0],fifteenmat,s=4)
plt.scatter(sigmamat[1],fourteenmat,s=4)
plt.scatter(sigmamat[2],thirteenmat,s=4)
plt.scatter(sigmamat[3],twelvemat,s=4)
plt.scatter(sigmamat[4],elevenmat,s=4)
plt.scatter(sigmamat[5],tenmat,s=4)
plt.scatter(sigmamat[6],ninemat,s=4)
plt.scatter(sigmamat[7],eightmat,s=4)
plt.scatter(sigmamat[8],sevenmat,s=4)
plt.scatter(sigmamat[9],sixmat,s=4)
plt.scatter(sigmamat[10],fivemat,s=4)
plt.scatter(sigmamat[11],fourmat,s=4)
plt.scatter(sigmamat[12],threemat,s=4)
plt.scatter(sigmamat[13],twomat,s=4)
plt.scatter(sigmamat[14],onemat,s=4)
plt.xlabel('\u03C3')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 4)
plt.scatter(Emat[0],fifteenmat,s=4)
plt.scatter(Emat[1],fourteenmat,s=4)
plt.scatter(Emat[2],thirteenmat,s=4)
plt.scatter(Emat[3],twelvemat,s=4)
plt.scatter(Emat[4],elevenmat,s=4)
plt.scatter(Emat[5],tenmat,s=4)
plt.scatter(Emat[6],ninemat,s=4)
plt.scatter(Emat[7],eightmat,s=4)
plt.scatter(Emat[8],sevenmat,s=4)
plt.scatter(Emat[9],sixmat,s=4)
plt.scatter(Emat[10],fivemat,s=4)
plt.scatter(Emat[11],fourmat,s=4)
plt.scatter(Emat[12],threemat,s=4)
plt.scatter(Emat[13],twomat,s=4)
plt.scatter(Emat[14],onemat,s=4)
plt.xlabel('E')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42

plt.subplot(2, 3, 5)
plt.scatter(seasonstartmat[0],fifteenmat,s=4)
plt.scatter(seasonstartmat[1],fourteenmat,s=4)
plt.scatter(seasonstartmat[2],thirteenmat,s=4)
plt.scatter(seasonstartmat[3],twelvemat,s=4)
plt.scatter(seasonstartmat[4],elevenmat,s=4)
plt.scatter(seasonstartmat[5],tenmat,s=4)
plt.scatter(seasonstartmat[6],ninemat,s=4)
plt.scatter(seasonstartmat[7],eightmat,s=4)
plt.scatter(seasonstartmat[8],sevenmat,s=4)
plt.scatter(seasonstartmat[9],sixmat,s=4)
plt.scatter(seasonstartmat[10],fivemat,s=4)
plt.scatter(seasonstartmat[11],fourmat,s=4)
plt.scatter(seasonstartmat[12],threemat,s=4)
plt.scatter(seasonstartmat[13],twomat,s=4)
plt.scatter(seasonstartmat[14],onemat,s=4)
plt.xlabel('Season Start Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42



plt.subplot(2, 3, 6)
plt.scatter(seasonendmat[0],fifteenmat,s=4)
plt.scatter(seasonendmat[1],fourteenmat,s=4)
plt.scatter(seasonendmat[2],thirteenmat,s=4)
plt.scatter(seasonendmat[3],twelvemat,s=4)
plt.scatter(seasonendmat[4],elevenmat,s=4)
plt.scatter(seasonendmat[5],tenmat,s=4)
plt.scatter(seasonendmat[6],ninemat,s=4)
plt.scatter(seasonendmat[7],eightmat,s=4)
plt.scatter(seasonendmat[8],sevenmat,s=4)
plt.scatter(seasonendmat[9],sixmat,s=4)
plt.scatter(seasonendmat[10],fivemat,s=4)
plt.scatter(seasonendmat[11],fourmat,s=4)
plt.scatter(seasonendmat[12],threemat,s=4)
plt.scatter(seasonendmat[13],twomat,s=4)
plt.scatter(seasonendmat[14],onemat,s=4)
plt.xlabel('Season End Time')
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
plt.rcParams['pdf.fonttype'] = 42
plt.subplots_adjust(hspace=0.5)
plt.savefig('Rangpur2014ParameterValuesAfterEachIterationofABC.pdf',bbox_inches='tight',transparent = True)
plt.show()




fig = plt.figure()
plt.suptitle('Prior and Posterior after 15 Iterations of ABC for Each Parameter - Rangpur 2014')

plt.subplot(3, 2, 1)
plt.hist(betamat[14], density=1)
plt.xlabel('\u03B2')
plt.hist([0,1],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 2)
plt.hist(epsilonmat[14], density=1)
plt.xlabel('\u03B5')
plt.hist([0,0.01],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 3)
plt.hist(seasonstartmat[14], density=1)
plt.xlabel('Season Start Time')
plt.hist([250,364],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 4)
plt.hist(seasonendmat[14], density=1)
plt.xlabel('Season End Time')
plt.hist([0,150],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])

plt.subplot(3, 2, 5)
plt.hist(sigmamat[14], density=1)
plt.xlabel('\u03C3')
plt.hist([0,0.25],bins=1, density=1, ls='dashed', alpha=0.5)
plt.rcParams['pdf.fonttype'] = 42
frame1 = plt.gca()
frame1.axes.yaxis.set_ticklabels([])
bbox=pylab.gca().get_position()
fig.subplots_adjust(hspace=0.5)
pylab.gca().set_position([bbox.x0 + 0.21, bbox.y0-0.03, bbox.x1-bbox.x0, bbox.y1-bbox.y0])

plt.savefig('Rangpur2014PriorandPosteriorafter15IterationsofABCforEachParameter.pdf',bbox_inches='tight',transparent = True)
plt.show()













































'''


fig = plt.figure()
plt.suptitle('Plots of Each Parameter against all other Parameters')

plt.subplot(2, 3, 1)
plt.scatter(betamat[0]4,epsilonmat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.ylim(min(epsilonmat14),max(epsilonmat14))
plt.xlabel('\u03B2')
plt.ylabel('\u03B5')

plt.subplot(2, 3, 2)
plt.scatter(betamat[0]4,sigmamat14,s=4)
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
plt.xlabel('\u03B2')
plt.ylabel('\u03C3')

plt.subplot(2, 3, 3)
plt.scatter(betamat[0]4,Emat14,s=4)
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