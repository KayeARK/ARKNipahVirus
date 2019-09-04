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

for i in range(1):

    N=1500 #population of village
    T=500 #elapsed time

    t=50  

    betamat = [0.00687601, 0.01593711, 0.00680132, 0.00359302, 0.00752853, 0.00626059, 0.00804918, 0.017491, 0.02005158, 0.00650009, 0.01085368, 0.02101174, 0.00859986, 0.01476881, 0.01170145, 0.01717267, 0.00884765, 0.00779128, 0.01176976, 0.02176349, 0.00326416, 0.0140602, 0.0164711, 0.00989381, 0.00518014, 0.01254709, 0.00990541, 0.00506421, 0.01024368, 0.01922609, 0.01414529, 0.01411674, 0.00235641, 0.01948095, 0.02068883, 0.00782334, 0.02170498, 0.00688367, 0.00550688, 0.01550287, 0.01606853, 0.00963604, 0.00793789, 0.0035317, 0.0019231, 0.00959194, 0.00941855, 0.01089694, 0.02285205, 0.01122994, 0.01426989, 0.00780663, 0.01053765, 0.00997899, 0.00477459, 0.00891857, 0.01119575, 0.02246546, 0.01993599, 0.00568708, 0.01231508, 0.00833311, 0.0137947, 0.00804834, 0.00963554, 0.0159798, 0.01552989, 0.00692738, 0.00414025, 0.01050585, 0.00415252, 0.00712945, 0.02087649, 0.00749801, 0.01563833, 0.01564794, 0.02096981, 0.01470828, 0.0058931, 0.00528879, 0.01725101, 0.01320732, 0.00552306, 0.0098778, 0.01609361, 0.0049472, 0.01244069, 0.00336715, 0.00890182, 0.01471051, 0.01792403, 0.01096086, 0.01403817, 0.00750394, 0.01562318, 0.02102519, 0.0211788, 0.00860091, 0.00582081, 0.00204879, 0.00600275, 0.00469071, 0.00936699, 0.00836088, 0.00648873, 0.0102914, 0.0073911, 0.0110788, 0.01007118, 0.0046361, 0.0120226, 0.00792003, 0.00174543, 0.00882999, 0.00749234, 0.0088929, 0.01514988, 0.01766558, 0.0156113, 0.00159629, 0.00712525, 0.02106483, 0.01201888, 0.00645009, 0.00770783, 0.00570676, 0.01107994, 0.01594993, 0.01549497, 0.01178459, 0.00514895, 0.0136107, 0.02009526, 0.00602881, 0.01949186, 0.00913739, 0.01054005, 0.0030187, 0.0058678, 0.00799378, 0.01628625, 0.00445623, 0.0107481, 0.00900234, 0.01398447, 0.00560993, 0.00436732, 0.02105049, 0.00546344, 0.01920403, 0.01456656, 0.01529357, 0.00255912, 0.02015456, 0.00712702, 0.01417152, 0.01690957, 0.00733687, 0.0165964, 0.00566199, 0.01113244, 0.01352785, 0.01748308, 0.01845079, 0.02094127, 0.01312817, 0.00386647, 0.00909846, 0.01926466, 0.01592424, 0.00827508, 0.00582268, 0.01363312, 0.00523046, 0.00454152, 0.01151966, 0.01133757, 0.00734058, 0.01461938, 0.01774812, 0.02171472, 0.00816867, 0.00406053, 0.01517765, 0.00524145, 0.01535369, 0.0115857, 0.020308, 0.00437656, 0.00740535, 0.00459049, 0.0012644, 0.0126982, 0.01044321, 0.01176027, 0.00629673, 0.00706719, 0.01789319, 0.00210149, 0.00319727]
    epsilonmat = [0.00032461, 0.00030359, 0.00028496, 0.00034118, 0.00031483, 0.00030371, 0.00031496, 0.00030188, 0.00028601, 0.00030571, 0.0003209, 0.00029949, 0.00033069, 0.00028914, 0.00029072, 0.00030372, 0.0002767, 0.00031936, 0.00033681, 0.00026078, 0.00033192, 0.00027533, 0.00031592, 0.0003375, 0.00028984, 0.00030579, 0.00029272, 0.00027939, 0.00030607, 0.00027433, 0.00032186, 0.0002762, 0.00031394, 0.00027867, 0.0003117, 0.00026473, 0.00029187, 0.00029817, 0.00026935, 0.00029101, 0.00030571, 0.00027524, 0.0003159, 0.0002943, 0.00029408, 0.00028987, 0.00032446, 0.00032074, 0.00030372, 0.00030684, 0.0003004, 0.00029414, 0.00027917, 0.00030706, 0.00032228, 0.00032291, 0.00029562, 0.00032395, 0.00026623, 0.00032757, 0.00027873, 0.00029807, 0.00031875, 0.00030698, 0.00028308, 0.00030176, 0.00032187, 0.00028555, 0.00028872, 0.00030459, 0.00032342, 0.00030658, 0.00028642, 0.00033043, 0.00026789, 0.00033707, 0.00030164, 0.00030909, 0.000322, 0.00027301, 0.0002887, 0.00028201, 0.00028642, 0.00032282, 0.00034279, 0.00031752, 0.00027625, 0.00030235, 0.00030558, 0.0002872, 0.0003196, 0.00031332, 0.00027188, 0.00029185, 0.0003196, 0.00029914, 0.00028446, 0.00026689, 0.00029998, 0.00029337, 0.00030042, 0.00030937, 0.00029214, 0.00029999, 0.00031809, 0.00033353, 0.00029981, 0.00030137, 0.0002757, 0.00030572, 0.00029087, 0.00028865, 0.00034061, 0.00029096, 0.00033739, 0.00026372, 0.00030422, 0.00027211, 0.00030487, 0.00027006, 0.00029925, 0.00033321, 0.00032066, 0.00028203, 0.00031352, 0.00032242, 0.00032442, 0.00031372, 0.00030655, 0.00027854, 0.00033656, 0.00030496, 0.00027211, 0.00030627, 0.00030194, 0.00030008, 0.00028735, 0.00031808, 0.00032344, 0.0003043, 0.00026699, 0.00026387, 0.00031159, 0.00032587, 0.0002925, 0.00030209, 0.00033697, 0.00034136, 0.00032284, 0.000336, 0.00032478, 0.00032116, 0.00030686, 0.00028246, 0.0002671, 0.00030318, 0.00029498, 0.00033348, 0.00028982, 0.00033659, 0.00031177, 0.00030223, 0.00032385, 0.00032056, 0.00029235, 0.00029168, 0.00030071, 0.0003395, 0.00029855, 0.00032186, 0.00029669, 0.00026511, 0.00032544, 0.00033179, 0.00032095, 0.00032281, 0.00032036, 0.00032341, 0.00033631, 0.00029613, 0.00028046, 0.00029114, 0.00028117, 0.00030273, 0.00032458, 0.00032933, 0.00032293, 0.00031193, 0.00033403, 0.00028174, 0.00031336, 0.0003229, 0.00027361, 0.00034037, 0.00030723, 0.00032121, 0.00032225, 0.00032463, 0.00028309, 0.00029515]
    sigmamat = [0.53604988, 0.20404651, 0.17151122, 0.17635798, 0.59271481, 0.19877865, 0.17265001, 0.38420702, 0.65956787, 0.27900419, 0.33803533, 0.31800318, 0.49890533, 0.21335655, 0.65520608, 0.15305551, 0.24371716, 0.66184528, 0.26489754, 0.18405315, 0.27738764, 0.26017961, 0.28249353, 0.49913939, 0.26646436, 0.67069308, 0.25653608, 0.26599461, 0.2815595, 0.2339708, 0.23820563, 0.50497299, 0.22207252, 0.32057044, 0.50982802, 0.25968816, 0.19337271, 0.46172364, 0.23694419, 0.32023888, 0.60344504, 0.23047454, 0.21821557, 0.23447374, 0.32503912, 0.27053878, 0.29708214, 0.30368583, 0.26279358, 0.61960733, 0.17184194, 0.23567056, 0.48299943, 0.25047719, 0.23763008, 0.48755079, 0.22706922, 0.29182214, 0.18643392, 0.30490322, 0.23018618, 0.54801438, 0.26155724, 0.55762398, 0.2582326, 0.53285561, 0.24989017, 0.24583828, 0.26878423, 0.2694151, 0.17070207, 0.1197321, 0.20537341, 0.16063166, 0.11827964, 0.30041659, 0.27275196, 0.2113283, 0.30129944, 0.63428652, 0.60067863, 0.18055092, 0.65251881, 0.54694258, 0.18981262, 0.25504148, 0.32550015, 0.66373531, 0.17922681, 0.28223602, 0.17030763, 0.49905291, 0.61660417, 0.29610227, 0.6283866, 0.16735381, 0.64874814, 0.51851712, 0.67165202, 0.17192937, 0.26860273, 0.25968262, 0.27908657, 0.33629931, 0.21959899, 0.17663573, 0.24373145, 0.22788564, 0.2869573, 0.34618571, 0.27452559, 0.49799826, 0.18774293, 0.28301979, 0.22551464, 0.3366914, 0.22758463, 0.21235863, 0.64507076, 0.30414859, 0.13907407, 0.21349365, 0.26196018, 0.24028667, 0.2106632, 0.66887726, 0.62801622, 0.29877205, 0.29258392, 0.26859883, 0.17627284, 0.47837011, 0.19016307, 0.33851338, 0.25451716, 0.2565522, 0.30399029, 0.34759029, 0.23416077, 0.25666264, 0.31654118, 0.18105782, 0.54371587, 0.57525253, 0.4878308, 0.25048183, 0.23908624, 0.51421633, 0.202421, 0.23117437, 0.25459329, 0.6378113, 0.20127282, 0.64971597, 0.27473233, 0.16936392, 0.32968631, 0.24753303, 0.16115552, 0.24016414, 0.18026991, 0.60987242, 0.51965035, 0.64277413, 0.25542403, 0.5506542, 0.49062138, 0.28799097, 0.3260081, 0.48038246, 0.28655548, 0.19861507, 0.46825547, 0.22017228, 0.24914186, 0.1519035, 0.23547588, 0.67183068, 0.15319229, 0.4777143, 0.63583594, 0.32220038, 0.49033774, 0.21494161, 0.21386177, 0.28500872, 0.13266482, 0.33231802, 0.25741334, 0.51056538, 0.17448398, 0.30542303, 0.33612932, 0.27443558, 0.25125054, 0.52492983, 0.25577813, 0.25379458, 0.24524964, 0.2192023]
    msemat = [29.068883707497267, 26.627053911388696, 27.892651361962706, 25.514701644346147, 28.583211855912904, 28.407745422683583, 29.154759474226502, 29.03446228191595, 26.77685567799177, 29.103264421710495, 26.38181191654584, 26.570660511172846, 28.30194339616981, 26.0, 28.178005607210743, 25.238858928247925, 28.687976575562104, 28.722813232690143, 27.568097504180443, 28.75760768909681, 27.367864366808018, 27.51363298439521, 25.84569596664017, 23.40939982143925, 28.26658805020514, 27.640549922170507, 28.053520278211074, 28.337254630609507, 28.142494558940577, 26.551836094703507, 24.269322199023193, 24.919871588754223, 27.712812921102035, 28.965496715920477, 28.809720581775867, 27.03701166919155, 27.92848008753788, 23.895606290697042, 27.892651361962706, 28.583211855912904, 24.758836806279895, 25.729360660537214, 29.103264421710495, 28.53068523537421, 23.895606290697042, 23.93741840717165, 27.54995462791182, 28.74021572639983, 28.513154858766505, 28.879058156387302, 20.97617696340303, 29.120439557122072, 22.869193252058544, 29.120439557122072, 28.42534080710379, 23.021728866442675, 28.495613697550013, 23.790754506740637, 27.0, 28.319604517012593, 28.24889378365107, 28.91366458960192, 26.814175355583846, 26.457513110645905, 27.018512172212592, 27.92848008753788, 27.184554438136374, 27.92848008753788, 28.600699292150182, 28.653097563788805, 28.071337695236398, 26.30589287593181, 28.635642126552707, 28.442925306655784, 28.495613697550013, 26.94438717061496, 25.84569596664017, 28.75760768909681, 26.1725046566048, 22.06807649071391, 24.43358344574123, 20.85665361461421, 24.71841418861655, 27.349588662354687, 27.92848008753788, 27.85677655436824, 23.08679276123039, 26.457513110645905, 26.19160170741759, 23.643180835073778, 26.30589287593181, 26.229754097208, 28.21347195933177, 27.92848008753788, 28.6705423736629, 28.372521918222215, 28.24889378365107, 23.558437978779494, 26.589471600616662, 28.89636655359978, 29.03446228191595, 27.202941017470888, 28.653097563788805, 29.086079144497972, 29.017236257093817, 24.269322199023193, 25.199206336708304, 25.826343140289914, 26.981475126464083, 27.055498516937366, 24.49489742783178, 28.53068523537421, 29.068883707497267, 23.515952032609693, 27.76688675382964, 23.853720883753127, 26.1725046566048, 27.83882181415011, 25.25866188063018, 27.40437921208944, 28.600699292150182, 28.653097563788805, 28.861739379323623, 24.61706725018234, 26.962937525425527, 24.839484696748443, 28.77498913987632, 28.827070610799147, 25.0, 28.319604517012593, 28.653097563788805, 27.110883423451916, 25.099800796022265, 27.16615541441225, 26.962937525425527, 29.189039038652847, 28.319604517012593, 28.284271247461902, 25.41653005427767, 27.80287754891569, 29.120439557122072, 29.086079144497972, 22.561028345356956, 23.108440016582687, 27.386127875258307, 27.910571473905726, 28.844410203711913, 26.38181191654584, 27.94637722496424, 27.349588662354687, 26.0, 28.948229652260256, 27.676705006196094, 26.92582403567252, 29.154759474226502, 27.258026340878022, 26.832815729997478, 28.77498913987632, 24.657656011875904, 28.319604517012593, 27.202941017470888, 28.687976575562104, 27.83882181415011, 23.043437243605826, 28.42534080710379, 27.073972741361768, 26.153393661244042, 27.85677655436824, 27.477263328068172, 28.89636655359978, 26.1725046566048, 27.40437921208944, 26.70205984563738, 26.870057685088806, 24.779023386727733, 27.477263328068172, 26.267851073127396, 26.1725046566048, 27.622454633866266, 28.948229652260256, 26.038433132583073, 28.478061731796284, 29.103264421710495, 28.407745422683583, 26.851443164195103, 28.39013913315678, 28.26658805020514, 28.635642126552707, 27.110883423451916, 28.39013913315678, 23.388031127053, 28.91366458960192, 29.0516780926679, 22.271057451320086, 28.687976575562104, 28.583211855912904, 28.965496715920477, 22.181073012818835, 26.77685567799177, 26.095976701399778]
    Emat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        
        
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

    mu1=((1/16)/(7/9))-(1/16)
    mu2=1/16
    mu=1/(365*67) #natural birth and death rate
    d=10
    gamma=10
    p=1
    E=random.choice(Emat) #intial number of infected individuals
    S=N-E #number of susceptibles
    I=1 #number of exposed
    R=0 #number of dead
    Q=0
    V=0
    inputSEIVQR = [S, E, I, V, Q, R]

    SEIVQR = []
    SEIVQR.append([t, S, E, I, V, Q, R]) #array of SEIR values

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
                
        N = S + E + I + V + Q
        rate1 = (beta * I * S)/N #human to human transmission
        rate2 = sigma * E #move from exposed to infected
        rate3 = mu2 * I #disease induced death
        rate4 = mu1 * I #recovery from the disease and move back to susceptible
        rate5 = mu * N + mu2*I #birth rate
        rate6 = mu * S #natural death from susceptible class
        rate7 = mu * E #natural death from exposed class
        rate8 = mu * I #natural death from infectious class
        rate9 = _epsilon * S #transmission from bats
        rate10 = d * I
        rate11 = mu1 * Q
        rate12 = mu2 * Q
        rate13 = mu * Q
        rate14 = (p * gamma * S)/N
        rate15 = mu * V
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
            
        elif sum(ratemat [:8])/ratetotal < r and r < sum(ratemat [:9])/ratetotal:
            S = S - 1
            E = E + 1
            ifb = ifb + 1
            
        elif sum(ratemat [:9])/ratetotal < r and r < sum(ratemat [:10])/ratetotal:
            I = I - 1
            Q = Q + 1
            
        elif sum(ratemat [:10])/ratetotal < r and r < sum(ratemat [:11])/ratetotal:
            Q = Q - 1
            S = S + 1
            
        elif sum(ratemat [:11])/ratetotal < r and r < sum(ratemat [:12])/ratetotal:
            Q = Q - 1
            R = R + 1

        elif sum(ratemat [:12])/ratetotal < r and r < sum(ratemat [:13])/ratetotal:
            Q = Q - 1
            
        elif sum(ratemat [:13])/ratetotal < r and r < sum(ratemat [:14])/ratetotal:
            S = S - 1
            V = V + 1
            
        else:
            V = V - 1

                    
        SEIVQR.append([t, S, E, I, V, Q, R]) #adds data to SIR matrix

                
    time = [row[0] for row in SEIVQR] #times to be plotted
    susceptible = [row[1] for row in SEIVQR] #susceptible individuals to be plotted
    exposed = [row[2] for row in SEIVQR]
    infected = [row[3] for row in SEIVQR] #infectious individuals to be plotted
    vaccinated = [row[4] for row in SEIVQR]
    quarantined = [row[5] for row in SEIVQR]
    dead = [row[6] for row in SEIVQR] #dead individuals to be plotted

    plt.step(time, exposed, 'b', label='Exposed')
    plt.step(time, infected, 'y', label='Infected')
    plt.step(time, dead, 'r', label='Dead')
    plt.step(time, quarantined, 'g', label='Quarantined')
    plt.step(time, vaccinated, 'm', label='Vaccinated')
            
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

colors = ['blue', 'yellow', 'red', 'green', 'magenta']
lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors]
labels = ['Exposed', 'Infectious', 'Dead', 'Quarantined', 'Vaccinated']
plt.legend(lines, labels)

def dSEIVQR_dt(X, t):
    return [-epsilon*X[0]-(beta*X[0]*X[2]/N)+mu1*X[2]+mu*(X[1]+X[2])+mu2*X[2]+mu1*X[3]-(p*gamma*X[0])/N,epsilon*X[0]+(beta*X[0]*X[2]/N)-sigma*X[1]-mu*X[1], sigma*X[1]-X[2]*(mu1+mu2+mu)-d*X[2],d*X[2]-X[3]*(mu1+mu2+mu),mu2*(X[2]+X[3]), (p*gamma*X[0])/N-mu*X[3]]
tsolve=np.linspace(50, list(t)[0], 1000)

dSEIVQR = odeint(dSEIVQR_dt, inputSEIVQR, tsolve)
plt.plot(tsolve, [row[1] for row in dSEIVQR], '--', c='k')
plt.plot(tsolve, [row[2] for row in dSEIVQR], '--', c='k')
plt.plot(tsolve, [row[3] for row in dSEIVQR], '--', c='k')
plt.plot(tsolve, [row[4] for row in dSEIVQR], '--', c='k')
plt.plot(tsolve, [row[5] for row in dSEIVQR], '--', c='k')


plt.show()

