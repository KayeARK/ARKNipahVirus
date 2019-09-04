import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math
import seaborn as sns; sns.set()
import scipy.stats as stats
from scipy.stats import gamma

df = pd.read_excel (r'C:\Users\Alex\Desktop\URSS Project\Datasets.xlsx', sheet_name='Faridpur2004')
    

def SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E):
    global cumifmat
    global itaumat
    global cumdmat
    global dtaumat
    T=1000 #elapsed time
    itau=startt
    dtau=ddays[0]
    I=0 #intial number of infected individuals
    S=N-I #number of susceptibles
    R=0 #number of dead
    ift=I
    d=deaths[0]
    mu=1/(365*67) #natural birth and death rate
    itaumat=[startt]
    dtaumat=[ddays[0]]
    cumifmat=[I]
    cumdmat=[d]

    
    while t < T:
        
        if I > 100:
            break
                 
        if t%365<304 and t%365>120:
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
            d = d + 1
            dtau=math.floor(t)
            dtaumat.append(dtau)
            cumdmat.append(d)
                        
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


betamat=[]
epsilonmat=[]
sigmamat=[]
#mu1mat=[]
#mu2mat=[]
#Emat=[]
msemat=[]

mu1=((1/16)/(7/9))-(1/16)
mu2=1/16

iterations=1

for b in range(iterations):

    mse=1000

    while mse>120:
        
        idays=list(range(-62, 50))
        ddays=[]
        cases=[0] * 112
        deaths=[]

        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            ddays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])
            deaths.append(df.loc[i, 'Cumulative number of deaths'])
            
            
        E=0 #random.randint(0,cases[-1])
        startt=-62
        t=startt    #idays[0]
        N=df.loc[0,'Population']
        beta=random.uniform(0,0.2)
        epsilon=random.uniform(0,0.0006)
        sigma=random.uniform(0,1/4)
        mu1=((1/16)/(7/9))-(1/16)
        mu2=1/16

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E)


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
#        deathserror=math.sqrt(np.sum((np.subtract(deaths, cumdmat))**2))
        mse=caseserror #+ deathserror

    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
#    mu1mat.append(mu1)
#    mu2mat.append(mu2)
 #   Emat.append(E)
    msemat.append(mse)

    plt.scatter(idays, cases, c='k', s=2)
    plt.scatter(itaumat, cumifmat, c='b', s=2)
    plt.title("Cumulative Cases")

plt.show()

'''            
        i=0
        
        while i < len(dtaumat)-1:
            if dtaumat[i]==dtaumat[i+1]:
                del cumdmat[i]
                del dtaumat[i]                
                                          
            elif dtaumat[i]+1!=dtaumat[i+1]:
                dtaumat.insert(i+1, dtaumat[i]+1)
                cumdmat.insert(i+1, cumdmat[i])

            else:
                i = i + 1                             
                                              
        while len(dtaumat)<len(ddays):
            dtaumat.append(dtaumat[-1]+1)
            cumdmat.append(cumdmat[-1])        
       
        while len(dtaumat)>len(ddays):
            ddays.append(ddays[-1]+1)
            deaths.append(deaths[-1])           
'''
            



print('betamat1','=',list(betamat))
print('epsilonmat1','=',list(epsilonmat))
print('sigmamat1','=',list(sigmamat))
#print('mu1mat1','=',mu1mat)
#print('mu2mat1','=',mu2mat)
print('msemat1','=',list(msemat))
#print('Emat1','=',list(Emat))
   


'''
betamat = [0.01391214, 0.00525139, 0.01616028, 0.00312683, 0.00461831, 0.00529692, 0.00239764, 0.00280137, 0.00778458, 0.0216832, 0.00428588, 0.00695388, 0.00194444, 0.00270418, 0.00469499, 0.00532848, 0.00519432, 0.00254117, 0.00631667, 0.00728097, 0.00243349, 0.01041578, 0.00346999, 0.00749248, 0.0170037, 0.01040085, 0.00482497, 0.00612504, 0.00622983, 0.00455272, 0.02063914, 0.00728622, 0.00501899, 0.00942407, 0.00668578, 0.01198445, 0.01212942, 0.00482605, 0.00215308, 0.00224512, 0.00707411, 0.00211906, 0.0163155, 0.0099457, 0.00438333, 0.01103382, 0.00652102, 0.00587815, 0.00270956, 0.00295655, 0.00756907, 0.01177687, 0.01783062, 0.00316758, 0.00306717, 0.01860794, 0.00939594, 0.00667833, 0.0060416, 0.01798398, 0.00121, 0.0034304, 0.00796371, 0.00489631, 0.01145587, 0.00128503, 0.00726878, 0.00391283, 0.00667516, 0.02024252, 0.02298734, 0.00246218, 0.00643281, 0.0060872, 0.00509293, 0.00385219, 0.00458703, 0.02278314, 0.00224174, 0.00290033, 0.0082801, 0.00365656, 0.00656413, 0.00591026, 0.00524782, 0.0115705, 0.00382844, 0.00260697, 0.00918048, 0.00617954, 0.00259238, 0.00344044, 0.0048134, 0.0067879, 0.00171171, 0.00884858, 0.00547949, 0.01091993, 0.00809856, 0.00211446, 0.00875258, 0.00925187, 0.00757257, 0.00755459, 0.0024706, 0.01639921, 0.01691466, 0.01171677, 0.00425902, 0.01752626, 0.00732141, 0.00733422, 0.01234347, 0.02020612, 0.01497148, 0.00255697, 0.00335499, 0.00993087, 0.00514631, 0.00943643, 0.00460117, 0.00800714, 0.00296194, 0.00125956, 0.00228832, 0.00669823, 0.00674101, 0.00526256, 0.00541044, 0.00134561, 0.02238544, 0.01139453, 0.00219548, 0.00806569, 0.00127846, 0.00179263, 0.00157137, 0.00357082, 0.00647112, 0.00500395, 0.02283667, 0.01270139, 0.00533015, 0.01377044, 0.00621276, 0.00221881, 0.00219513, 0.00237282, 0.00275111, 0.00322144, 0.00651309, 0.00168672, 0.00348205, 0.01156053, 0.00599256, 0.00400982, 0.00399679, 0.00219905, 0.00519358, 0.0063128, 0.01322388, 0.01125162, 0.02304456, 0.00288905, 0.00584016, 0.00692162, 0.00215366, 0.00545528, 0.0021333, 0.00856707, 0.0230997, 0.00508732, 0.00602631, 0.00538707, 0.00137847, 0.00307292, 0.00606248, 0.01001133, 0.01210877, 0.00473333, 0.00580663, 0.00503843, 0.00408652, 0.00399434, 0.0083356, 0.00416563, 0.00409031, 0.00481895, 0.02040784, 0.00265218, 0.00696577, 0.00151231, 0.00965298, 0.01994474, 0.00350962, 0.00574676, 0.00464248, 0.00314965, 0.00599696, 0.00916308]
epsilonmat = [0.00029187, 0.00030477, 0.00031583, 0.00029277, 0.00029528, 0.00029268, 0.00034852, 0.00030434, 0.0002996, 0.000289, 0.00028099, 0.00032432, 0.00028631, 0.00030021, 0.00028149, 0.00028723, 0.00031082, 0.00030471, 0.00028915, 0.00029847, 0.00027455, 0.0003375, 0.0003069, 0.0002964, 0.00027114, 0.00030508, 0.00030054, 0.00029755, 0.00033316, 0.00030637, 0.00028571, 0.00029647, 0.00031505, 0.00028939, 0.00028456, 0.0002744, 0.0002971, 0.00028532, 0.00031859, 0.00030302, 0.00034702, 0.00030771, 0.00029772, 0.00031168, 0.00028329, 0.00029056, 0.00026267, 0.00030293, 0.00029967, 0.00030495, 0.00027558, 0.00028144, 0.00029596, 0.00028764, 0.00030304, 0.0002837, 0.00028973, 0.0003056, 0.00029568, 0.00029172, 0.00034693, 0.00027545, 0.00029148, 0.00031024, 0.00031275, 0.00030644, 0.00031541, 0.00031124, 0.00030135, 0.0002788, 0.00030873, 0.00029832, 0.00029586, 0.00030111, 0.00028859, 0.0003145, 0.00030026, 0.00030191, 0.00031064, 0.00027892, 0.00032788, 0.00029052, 0.00031035, 0.00031413, 0.00029487, 0.00031367, 0.00030831, 0.00029203, 0.00030161, 0.00029894, 0.0003282, 0.00023857, 0.00029869, 0.00027862, 0.00027293, 0.00031578, 0.00030617, 0.00029429, 0.00029386, 0.00029227, 0.00029504, 0.00030389, 0.00029091, 0.00029146, 0.00031117, 0.00028682, 0.00033324, 0.00028804, 0.00026699, 0.00027492, 0.00029653, 0.00029749, 0.00029525, 0.00028985, 0.00029217, 0.00029191, 0.00030872, 0.00033801, 0.00029041, 0.00029582, 0.00027298, 0.00028531, 0.00033569, 0.00032748, 0.00028231, 0.00033569, 0.00024441, 0.00030295, 0.00028611, 0.00031192, 0.00029951, 0.00031304, 0.00031247, 0.00028818, 0.00028528, 0.00029933, 0.00029098, 0.00032796, 0.00028209, 0.00034182, 0.00029826, 0.00031022, 0.00031495, 0.0002819, 0.00030414, 0.000338, 0.00031513, 0.00028951, 0.00033199, 0.00031531, 0.00029868, 0.00027999, 0.00030899, 0.00030046, 0.00029477, 0.00031233, 0.00033191, 0.00029803, 0.00029242, 0.00029855, 0.0002927, 0.00025267, 0.00026823, 0.00029837, 0.0002983, 0.0002983, 0.00029003, 0.00029693, 0.00030121, 0.00030527, 0.00030113, 0.00029162, 0.00033426, 0.00031089, 0.0003146, 0.00028566, 0.00028527, 0.0002742, 0.00029619, 0.00027431, 0.00030011, 0.00028727, 0.00033983, 0.000334, 0.0002998, 0.00029135, 0.00028882, 0.00028734, 0.00030144, 0.00029347, 0.00029352, 0.00030201, 0.00029732, 0.00030698, 0.00028101, 0.00033642, 0.00028381, 0.00030282, 0.00029897, 0.000315]
sigmamat = [0.15156599, 0.11952697, 0.12746737, 0.16178062, 0.17190457, 0.15801849, 0.11938043, 0.16479517, 0.12774386, 0.17853204, 0.17188538, 0.17128957, 0.23185107, 0.21685045, 0.14998871, 0.21301258, 0.11449059, 0.16926901, 0.12804851, 0.18062174, 0.17264397, 0.22508902, 0.19613365, 0.1322886, 0.13630199, 0.19361617, 0.16074231, 0.17766074, 0.14427803, 0.2183349, 0.23363052, 0.15235387, 0.17650228, 0.17860647, 0.17064208, 0.19851838, 0.23416448, 0.16726481, 0.16074565, 0.16974788, 0.14814376, 0.16960102, 0.13025205, 0.23533478, 0.17163428, 0.1859047, 0.13924963, 0.23077714, 0.1775138, 0.12910732, 0.17515379, 0.22463805, 0.1712555, 0.21115, 0.16443582, 0.18115804, 0.13076525, 0.22569203, 0.11790337, 0.11439636, 0.18080905, 0.1879373, 0.22874935, 0.23579005, 0.15052828, 0.17868629, 0.17933684, 0.14617004, 0.22551777, 0.14575486, 0.16098189, 0.15480053, 0.18971334, 0.16363067, 0.19956593, 0.20067393, 0.15899169, 0.14029002, 0.17189249, 0.21673052, 0.10809199, 0.1875453, 0.17861209, 0.16563173, 0.21407564, 0.12253106, 0.16767888, 0.15689646, 0.18666096, 0.1760736, 0.1606641, 0.19488671, 0.1346334, 0.13686746, 0.17828735, 0.12715366, 0.15850749, 0.14548007, 0.15495195, 0.15072828, 0.13439749, 0.16350187, 0.16470275, 0.16234268, 0.10458989, 0.15399837, 0.12892601, 0.17010563, 0.13078732, 0.16388039, 0.13646036, 0.1670597, 0.11867319, 0.15335999, 0.16523638, 0.1438134, 0.16322266, 0.13590984, 0.15861116, 0.21053626, 0.20471689, 0.15268696, 0.20395416, 0.18475453, 0.13148327, 0.1567492, 0.15783331, 0.22479151, 0.1446119, 0.15681835, 0.1727401, 0.15559922, 0.21298811, 0.13039439, 0.1481591, 0.20976857, 0.1560951, 0.15368216, 0.15251273, 0.13983765, 0.16523606, 0.16391814, 0.15816824, 0.11768007, 0.15640347, 0.16237007, 0.11318712, 0.16989851, 0.12908153, 0.17443428, 0.15598987, 0.16065341, 0.21115745, 0.12815082, 0.14742281, 0.15299689, 0.12532777, 0.12937177, 0.13122351, 0.12703694, 0.180352, 0.15641989, 0.18104811, 0.17021775, 0.15811767, 0.16418384, 0.20205195, 0.15974329, 0.17865734, 0.1729041, 0.1606288, 0.18708023, 0.14484922, 0.16538781, 0.17984659, 0.21152617, 0.19730278, 0.11306134, 0.19454553, 0.15160203, 0.16054232, 0.13007511, 0.13034489, 0.14576953, 0.16750988, 0.14421887, 0.21230437, 0.14570665, 0.14873451, 0.17863053, 0.22233105, 0.14827241, 0.17668834, 0.22138914, 0.1274101, 0.13067846, 0.16613096, 0.15595436, 0.15661775, 0.14702655]
msemat = [33.97057550292606, 33.91164991562634, 31.480152477394387, 32.984845004941285, 32.32645975048923, 32.60368077380221, 32.43454948045371, 32.58834147360065, 33.46640106136302, 33.0, 32.68026927673638, 33.66006535941367, 32.95451410656816, 31.606961258558215, 31.272991542223778, 32.09361307176243, 32.51153641401772, 34.132096331752024, 32.17141588429082, 33.97057550292606, 33.12099032335839, 22.891046284519195, 33.94112549695428, 31.304951684997057, 31.12876483254676, 22.44994432064365, 30.919249667480614, 32.848135411313685, 32.09361307176243, 29.8496231131986, 29.597297173897484, 31.480152477394387, 33.166247903554, 32.78719262151, 32.90896534380867, 33.13608305156178, 32.863353450309965, 25.11971337416094, 33.8673884437522, 34.08812109811862, 31.20897306865447, 31.73326330524486, 32.41913015489465, 31.448370387032774, 32.046840717924134, 32.01562118716424, 32.96968304366907, 33.51119216023208, 31.52776554086889, 27.0, 31.22498999199199, 29.765752132274432, 30.577769702841312, 34.0, 30.870698080866262, 33.85262175962151, 26.90724809414742, 33.97057550292606, 32.68026927673638, 32.63433774416144, 31.52776554086889, 33.63034344160047, 33.71943060017473, 33.015148038438355, 32.49615361854384, 33.57082066318904, 32.38826948140329, 30.347981810987037, 28.6705423736629, 32.64965543462902, 27.910571473905726, 31.144823004794873, 25.41653005427767, 28.19574435974337, 31.416556144810016, 33.36165463522455, 29.376861643136763, 31.12876483254676, 32.78719262151, 33.823069050575526, 30.919249667480614, 33.075670817082454, 29.03446228191595, 31.0322412983658, 30.967725134404045, 30.0, 24.61706725018234, 31.701734968294716, 28.948229652260256, 31.464265445104548, 28.722813232690143, 31.416556144810016, 33.63034344160047, 29.512709126747414, 27.964262908219126, 33.8673884437522, 30.757112998459398, 33.85262175962151, 33.66006535941367, 33.98529093593286, 34.0, 34.08812109811862, 32.984845004941285, 30.24896692450835, 31.622776601683793, 31.827660925679098, 33.67491648096547, 29.949958263743873, 33.60059523282288, 31.480152477394387, 32.87856444554719, 32.802438933713454, 32.07802986469088, 30.18277654557314, 29.916550603303182, 32.12475680841802, 33.98529093593286, 33.71943060017473, 28.77498913987632, 32.72613634390714, 32.01562118716424, 33.704599092705436, 33.37663853655727, 29.866369046136157, 30.789608636681304, 32.848135411313685, 28.77498913987632, 30.083217912982647, 32.863353450309965, 29.698484809834994, 32.69556544854363, 29.206163733020468, 34.14674215792775, 34.11744421846396, 33.391615714128, 33.04542328371661, 27.258026340878022, 33.481338085566414, 32.64965543462902, 32.31098884280702, 33.36165463522455, 33.645207682521445, 33.45145736735546, 33.704599092705436, 34.02939905434711, 33.71943060017473, 30.967725134404045, 32.802438933713454, 31.71750305430741, 33.25657829663178, 28.61817604250837, 30.298514815086232, 30.364452901377952, 31.0322412983658, 31.464265445104548, 31.984371183438952, 33.21144381083123, 31.0, 33.88214869219483, 31.144823004794873, 30.528675044947494, 32.848135411313685, 33.63034344160047, 32.4037034920393, 33.406586176980134, 30.56141357987225, 33.98529093593286, 34.0, 33.45145736735546, 30.854497241083024, 33.015148038438355, 32.60368077380221, 33.18132004607411, 30.166206257996713, 25.84569596664017, 30.967725134404045, 32.863353450309965, 33.03028912982749, 34.044089061098404, 32.046840717924134, 32.28002478313795, 34.0, 26.095976701399778, 31.63858403911275, 33.645207682521445, 31.76476034853718, 34.0, 33.421549934136806, 30.033314835362415, 32.68026927673638, 32.7414110874898, 33.075670817082454, 30.72458299147443, 33.704599092705436, 31.811947441173732, 32.357379374726875, 33.301651610693426, 33.896902513356586, 33.704599092705436, 30.805843601498726]
Emat = [0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
'''

betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
#mu1matABC=mu1mat
#mu2matABC=mu2mat
msematABC=msemat
#EmatABC=Emat


for c in range(50):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
#    mu1mat=[]
#    mu2mat=[]
 #   Emat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=list(range(-62, 50))
            ddays=[]
            cases=[0] * 112
            deaths=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                ddays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                deaths.append(df.loc[i, 'Cumulative number of deaths']) 
            
            
            

            
                                    
            t=startt
            N=df.loc[0,'Population']
            
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

#            histmu1,binsmu1 = np.histogram(mu1matABC)
#            bin_midpointsmu1 = binsmu1[:-1]+np.diff(binsmu1)/2
#            cdfmu1=np.cumsum(histmu1)
#            cdfmu1=cdfmu1 / cdfmu1[-1]
#            valuesmu1 = np.random.rand(1)
#            value_binsmu1 = np.searchsorted(cdfmu1,valuesmu1)
#            mu1 = bin_midpointsmu1[value_binsmu1] + random.uniform((binsmu1[0]-binsmu1[1])/2,(binsmu1[1]-binsmu1[0])/2)

#            histmu2,binsmu2 = np.histogram(mu2matABC)
#            bin_midpointsmu2 = binsmu2[:-1]+np.diff(binsmu2)/2
#            cdfmu2=np.cumsum(histmu2)
#            cdfmu2=cdfmu2 / cdfmu2[-1]
#            valuesmu2 = np.random.rand(1)
#            value_binsmu2 = np.searchsorted(cdfmu2,valuesmu2)
#            mu2 = bin_midpointsmu2[value_binsmu2] + random.uniform((binsmu2[0]-binsmu2[1])/2,(binsmu2[1]-binsmu2[0])/2)

            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E)

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
#            deathserror=math.sqrt(np.sum((np.subtract(deaths, cumdmat))**2))
            mse=caseserror #+ deathserror



        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
#        mu1mat.append(mu1)
#        mu2mat.append(mu2)
#        Emat.append(E)
        msemat.append(mse)
        
        
       # plt.scatter(idays, cases, c='k', s=2)
       # plt.scatter(itaumat, cumifmat, c='b', s=2)
       # plt.title("Cumulative Cases")
   

    
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
#    mu1matABC=mu1mat
#    mu2matABC=mu2mat
    msematABC=msemat
 #   EmatABC=Emat

    print('betamat',c+2,'=',list(betamat))
    print('epsilonmat',c+2,'=',list(epsilonmat))
    print('sigmamat',c+2,'=',list(sigmamat))
#    print('mu1mat',c+2,'=',mu1mat)
#    print('mu2mat',c+2,'=',mu2mat)
    print('msemat',c+2,'=',list(msemat))
 #   print('Emat',c+2,'=',list(Emat))
    

'''               
            i=0
            
            while i < len(dtaumat)-1:
                if dtaumat[i]==dtaumat[i+1]:
                    del cumdmat[i]
                    del dtaumat[i]                
                                              
                elif dtaumat[i]+1!=dtaumat[i+1]:
                    dtaumat.insert(i+1, dtaumat[i]+1)
                    cumdmat.insert(i+1, cumdmat[i])

                else:
                    i = i + 1                             
                                                  
            while len(dtaumat)<len(ddays):
                dtaumat.append(dtaumat[-1]+1)
                cumdmat.append(cumdmat[-1])        
           
            while len(dtaumat)>len(ddays):
                ddays.append(ddays[-1]+1)
                deaths.append(deaths[-1])           

'''            


            
       


#plt.show()  

# HISTOGRAM #

#plt.hist(sigmamat, density=1)
#plt.show()