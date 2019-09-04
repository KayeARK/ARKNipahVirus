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
    

def SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, alpha, omega):
    global cumifmat
    global itaumat
    global cumdmat
    global dtaumat
    T=1000 #elapsed time
    itau=idays[0]
    dtau=ddays[0]
    I=cases[0] #intial number of infected individuals
    S=N-I #number of susceptibles
    R=0 #number of dead
    H=0
    ift=I
    d=deaths[0]
    mu=1/(365*67) #natural birth and death rate
    itaumat=[idays[0]]
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
                    
        N = S + E + I + H
        rate1 = (beta * I * S)/N #human to human transmission
        rate2 = (alpha * beta * H * S)/N
        rate3 = _epsilon * S
        rate4 = mu1 * H
        rate5 = mu1 * I
        rate6 = mu * N + mu2 * (I + H)
        rate7 = mu * S
        rate8 = mu * E
        rate9 = mu * I
        rate10 = mu * H
        rate11 = sigma * E
        rate12 = omega * I
        rate13 = mu2 * H
        rate14 = mu2 * I
        ratemat = [rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, rate10, rate11, rate12, rate13, rate14]
        ratetotal = sum(ratemat)
                    
        dt = -math.log(random.uniform(0,1))/ratetotal #timestep
        t = t + dt
                    
        r=random.uniform(0,1)
        
        if r < sum(ratemat [:1])/ratetotal: #move to infected class
            S = S - 1
            E = E + 1
                    
        elif sum(ratemat [:1])/ratetotal < r and r < sum(ratemat [:2])/ratetotal:
            S = S - 1
            E = E + 1
                    
        elif sum(ratemat [:2])/ratetotal < r and r < sum(ratemat [:3])/ratetotal:
            S = S - 1
            E = E + 1
                    
        elif sum(ratemat [:3])/ratetotal < r and r < sum(ratemat [:4])/ratetotal:
            H = H - 1
            S = S + 1
                    
        elif sum(ratemat [:4])/ratetotal < r and r < sum(ratemat [:5])/ratetotal:
            I = I - 1
            S = S + 1
                    
        elif sum(ratemat [:5])/ratetotal < r and r < sum(ratemat [:6])/ratetotal:
            S = S + 1
                    
        elif sum(ratemat [:6])/ratetotal < r and r < sum(ratemat [:7])/ratetotal:
            S = S - 1
                    
        elif sum(ratemat [:7])/ratetotal < r and r < sum(ratemat [:8])/ratetotal:
            E = E - 1
                    
        elif sum(ratemat [:8])/ratetotal < r and r < sum(ratemat [:9])/ratetotal:
            I = I - 1
                
        elif sum(ratemat [:9])/ratetotal < r and r < sum(ratemat [:10])/ratetotal:
            H = H - 1    
            
        elif sum(ratemat [:10])/ratetotal < r and r < sum(ratemat [:11])/ratetotal:
            E = E - 1
            I = I + 1
            ift = ift + 1
            itau=math.floor(t)
            itaumat.append(itau)
            cumifmat.append(ift)
            
        elif sum(ratemat [:11])/ratetotal < r and r < sum(ratemat [:12])/ratetotal:
            I = I - 1
            H = H + 1
            
        elif sum(ratemat [:12])/ratetotal < r and r < sum(ratemat [:13])/ratetotal:
            H = H - 1
            R = R + 1
            d = d + 1
            dtau=math.floor(t)
            dtaumat.append(dtau)
            cumdmat.append(d)           
                
        else:
            I = I - 1
            R = R + 1
            d = d + 1
            dtau=math.floor(t)
            dtaumat.append(dtau)
            cumdmat.append(d)            
       

betamat=[]
epsilonmat=[]
sigmamat=[]
#mu1mat=[]
#mu2mat=[]
Emat=[]
alphamat=[]
omegamat=[]
msemat=[]

iterations=200

for b in range(iterations):

    mse=1000

    while mse>100:
        
        idays=[]
        ddays=[]
        cases=[]
        deaths=[]

        for i in range(len(df.index)):
            idays.append(df.loc[i, 'Day'])
            ddays.append(df.loc[i, 'Day'])
            cases.append(df.loc[i, 'Cumulative number of cases'])
            deaths.append(df.loc[i, 'Cumulative number of deaths'])
            
            
        E=random.randint(0,cases[-1])
        t=idays[0]
        N=df.loc[0,'Population']
        beta=random.uniform(0,0.2)
        epsilon=random.uniform(0,0.0006)
        sigma=random.uniform(0,1)
        mu1=((1/16)/(7/9))-(1/16)
        mu2=1/16
        alpha=random.uniform(0,1)
        omega=random.uniform(0,100)

        SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, alpha, omega)


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

            

        caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
        deathserror=math.sqrt(np.sum((np.subtract(deaths, cumdmat))**2))
        mse=caseserror # + deathserror
        
   
    betamat.append(beta)
    epsilonmat.append(epsilon)
    sigmamat.append(sigma)
#    mu1mat.append(mu1)
#    mu2mat.append(mu2)
    Emat.append(E)
    msemat.append(mse)
    alphamat.append(alpha)
    omegamat.append(omega)
    
print('betamat1','=',betamat)
print('epsilonmat1','=',epsilonmat)
print('sigmamat1','=',sigmamat)
#print('mu1mat1','=',mu1mat)
#print('mu2mat1','=',mu2mat)
print('msemat1','=',msemat)
print('Emat1','=',Emat)
print('alphamat1','=',alphamat)
print('omegamat1','=',omegamat)



'''    
    plt.scatter(idays, cases, c='k', s=2)
    plt.scatter(itaumat, cumifmat, c='b', s=2)
    plt.title("Cumulative Cases")
    

    

    plt.scatter(ddays, deaths, c='k', s=2)
    plt.scatter(dtaumat, cumdmat, c='b', s=2)
    plt.title("Cumulative Deaths")
    
plt.show()
    
betamat=[14.937211880948592, 24.0641556103157, 19.020502713441118, 25.119950157310978, 37.96554528147449, 2.332422946973095, 39.14801374097518, 23.800344058016815, 25.899484225216764, 20.371402043913843, 13.159412759730932, 31.414498219262445, 13.517520707736502, 11.222723470877305, 29.748349095799647, 9.244308854322142, 49.180178797839595, 14.806114249055252, 7.0340663486543225, 17.257510196840293, 4.047234472342848, 32.23258140003835, 40.51764228561808, 40.86196197042262, 8.57632223835409, 30.595917181762005, 16.876572680228907, 48.215018433715656, 17.077712180620452, 11.521153965067278, 5.4654676070876205, 7.030830641629482, 9.80997437646376, 8.828386143126371, 18.308135429495376, 14.227964169405071, 6.337824192415459, 14.21112741257708, 33.03056807154276, 26.107634746871035, 21.186549758004343, 18.443036381568444, 15.135149826590139, 34.24669358667819, 18.387247741142193, 24.08191401069787, 14.298829188077805, 11.290482601577423, 14.190103129100878, 24.952303525412688, 4.372303793486237, 49.16648344937853, 25.909636046215983, 15.654254015862328, 15.09901989126078, 25.6119905188444, 41.6279287540475, 21.139222871510952, 39.636963917124675, 49.157041103314235, 11.852322136665066, 37.54748343302145, 44.635695065066564, 4.044199412087335, 12.12017277523167, 21.790933763450028, 25.408124727134574, 40.14453070335768, 30.162357961343076, 25.684629946540866, 41.79733481302717, 8.052856402689423, 22.272746681186582, 40.10314141569285, 30.06311742856873, 9.033651850963459, 41.342632404608004, 21.781230666659585, 44.694286554350555, 8.40977571170774, 20.03944466662621, 46.49985905108471, 23.630826478758042, 7.809609983450233, 44.518810025592884, 26.392423042954366, 22.714936652095513, 16.939345968037923, 10.648453109769696, 6.823252226019666, 9.438625390320116, 27.12763929178068, 19.977934285199627, 24.266918694490506, 14.26056566451529, 4.265836543511287, 23.706148125303887, 17.972197917894796, 49.3586549537312, 16.864152365451666]
epsilonmat=[0.00019345334221815825, 0.00015014388466009888, 0.0001944037021154882, 0.00023121693560800282, 0.00017435633224782587, 0.00034277393781285205, 0.00028889107980891116, 0.00029459943143223124, 0.00031845239836428876, 0.0002737020621374288, 0.00027381875459346004, 0.0001793511670697753, 0.00011122410105601033, 0.0002814582019510531, 0.0002753513838827136, 0.00019613315809145925, 9.900897154939437e-05, 0.00031141152465543443, 0.00044030051911067336, 0.00013813313728353626, 0.00024693749488615194, 0.00020261889537997468, 5.2521426228287264e-05, 0.0002473094869638958, 0.0003062501359365062, 5.812426440117791e-05, 0.00027407627748965804, 0.00011059617856119519, 0.0002285164484781317, 0.00029236500812952916, 0.0003225294474050422, 0.00028235579174969584, 0.0002548582823573953, 0.0003421768420510999, 0.0001667910311361408, 0.0001185290364688888, 0.0003146299035439799, 0.00028775657524122934, 5.4924313012496454e-05, 0.000166243179193311, 0.0001919385770404405, 0.00021633867388324502, 0.00013602551261486962, 0.00014032815171746883, 0.0001807808269314426, 0.00023143319818658947, 0.0002489206222321153, 0.00037468182270519723, 0.00037434226956753594, 0.0003852331394487709, 0.0002507534469271343, 0.00021925737350412467, 0.00019764181355450051, 0.00019082356571835902, 0.0002710126766026756, 0.0002226566691824532, 0.00019779364327354588, 0.00045305239107042193, 0.00012371866212986215, 0.0002083027825400947, 0.00026210061436828804, 0.00010718898569234714, 4.1651931818506014e-05, 0.0003234562114325691, 0.00021839016737166416, 7.957941814926461e-05, 7.747284105252461e-05, 7.253587686461249e-05, 0.0003293840690359862, 0.00017164209203662185, 0.000314428112523539, 0.00027862003265747784, 0.00034714235866437515, 0.0002118658308137309, 0.00020260270132411306, 0.0003687880813309056, 8.819084053587978e-05, 0.00010448706851700596, 0.00018095851676025443, 0.00045762231130547015, 0.0003584499560345136, 0.00039522573204667433, 0.00013149216903454464, 0.00034683763315577934, 0.0001474343455288596, 0.0003686867466602852, 0.00024699176271338253, 0.0002587412215439531, 0.0002121488896977276, 0.0001789215983352881, 0.00023414431964338567, 0.00014860269155831207, 0.0001616074747415991, 0.0002230678542299711, 8.580265123634979e-05, 0.00046496577038488763, 8.211186856874553e-05, 6.45890204343531e-05, 0.00011337620050960873, 0.00024770229109019625]
sigmamat=[0.25982506986400034, 0.12789844619031387, 0.16139527200829495, 0.29849583306062033, 0.1799644401335142, 0.19311666889976986, 0.20397971451617214, 0.08785212064698511, 0.08010252054062017, 0.1329930035681246, 0.10217983106239359, 0.19777245813439082, 0.15191424429669675, 0.2322069995294176, 0.21824712998197882, 0.22955557746513378, 0.12293640904999703, 0.09176681318862956, 0.08181904322512001, 0.18805669088280186, 0.13558347334637788, 0.20730983196139519, 0.1439160076822631, 0.14214691373363153, 0.1367552145751234, 0.20376567817896296, 0.0676678091390015, 0.12712519461477867, 0.11156476788188852, 0.07015234398055692, 0.11904547965714607, 0.13950276148308627, 0.17317006053153017, 0.17124320203585752, 0.08242425682246368, 0.14635173695725234, 0.1862381092682468, 0.11985271633492289, 0.18791407736563326, 0.10815952403726103, 0.13305803216655454, 0.27974732358399057, 0.14646329199182873, 0.26901403642188426, 0.19732236413344828, 0.11957501578125518, 0.14532808644726858, 0.09657300740971342, 0.2640652384570953, 0.18885371782141047, 0.3256602193345587, 0.1732445949704301, 0.1510599620847891, 0.31129029254622287, 0.394432374728791, 0.09278035191314038, 0.1688959987514279, 0.28120855053432947, 0.19270641592368187, 0.20047951603248337, 0.13933557649899753, 0.15083246292459485, 0.24822555911280653, 0.27755875930124696, 0.17791855163128878, 0.24810198592202803, 0.08081850051607098, 0.12558505816793275, 0.11629648039775498, 0.18631861137837435, 0.11737940267037517, 0.16551993414154476, 0.17647712896169654, 0.2990881459663738, 0.20967933879602485, 0.1880768740972023, 0.18744211348851936, 0.16723896573246444, 0.11830164589642145, 0.28212583545347536, 0.24959998576004339, 0.12054681516035726, 0.10062845580675694, 0.07354760434854579, 0.1336943606052603, 0.16789470191767863, 0.17447781444533172, 0.22812550963098466, 0.11925615241425758, 0.1643891585446181, 0.13686288987622253, 0.1216562441046024, 0.08496111967805697, 0.17789166765268916, 0.1407238888163116, 0.19086128858717222, 0.11056801134821848, 0.3947688423345814, 0.19759053615572542, 0.08371459003608883]
mu1mat=[3.7354437010167887, 46.17546909354143, 13.937092600808187, 41.50535640876619, 40.22102695804876, 9.219134356669406, 19.902934021159762, 16.833550877432124, 45.49619261454575, 42.89008392115725, 12.345968022754866, 31.075226639876952, 6.561864053352212, 4.920292502084111, 23.256674663809612, 18.68391891786368, 27.837652547483515, 9.849558654775386, 40.89418220041863, 17.93820583335502, 9.37472466965039, 23.205232356612438, 24.090204605814076, 15.535235867338216, 12.604361337045129, 8.239770729035667, 22.93782509606585, 18.511662829209037, 25.002617379655952, 12.466769547566209, 3.206074311393259, 10.954421872701264, 12.1488793209314, 27.92115400716798, 27.07373365270833, 16.39949720317449, 4.3043081061910735, 27.148905454499513, 3.593830663009406, 6.242574771709458, 34.16110061775466, 26.84760837049356, 14.78406301935467, 4.538118207007824, 25.166477732632984, 5.593613804956366, 18.894527102817378, 23.14224619064994, 36.710925297055525, 14.807423760168097, 5.607577721732671, 18.373544776001864, 26.721783674423865, 20.194044597441096, 7.658459903342763, 22.63490226323596, 20.798584053697137, 22.248838507275696, 25.472303186615576, 37.030899160828646, 14.085345536025134, 40.79076177410174, 36.914643114812456, 17.335398643389787, 26.66557126473296, 12.711974754745087, 1.8032684307264069, 17.860644414390542, 29.46352415446209, 7.251425049366, 10.775684548452414, 14.676383633608781, 15.559788362538491, 9.562034276259872, 44.1227323520088, 25.095330943518906, 42.63817023573412, 43.6307890281463, 8.353735704150573, 44.34070067169207, 44.189581330043396, 47.086676697161394, 13.390202375780902, 9.037185589650337, 20.505607308640748, 43.7568306625025, 22.072193251137058, 22.140882837817987, 27.40658313742898, 9.338036224160223, 19.93461954932629, 26.824147866918963, 26.686970926489508, 37.94601238685102, 1.9151649747101818, 6.328293156686332, 14.858638340048824, 5.05607667410361, 30.615548177734425, 11.172710639422927]
mu2mat=[23.596876484160667, 49.93665311929296, 38.97139073500759, 24.42184564818471, 30.1790632610953, 40.89063599846654, 44.91659052446578, 37.74560845926836, 45.85716320635137, 30.780502611884902, 27.039141143884255, 32.695958109267025, 23.265478487395747, 26.553263865271852, 42.023811480083836, 45.5086460854501, 41.7420244752138, 36.41129709473468, 49.93140398346076, 20.569892251315153, 39.87431704529684, 35.99822855585335, 41.29213010529614, 38.88412410407873, 30.0107697053633, 36.01180807468588, 42.78121686753261, 31.915881699265974, 35.876911623510246, 34.94379472250515, 47.23192351282188, 13.093302967029446, 14.586759473456157, 27.949469189130312, 38.692331437784894, 33.52983931321879, 44.926832992721714, 35.770096587058845, 25.826987905327808, 35.37121314114942, 28.581100669792686, 22.59780012566232, 17.689202743052075, 45.14905155056138, 32.7002374755328, 29.413004475025357, 44.62369630157122, 38.74942580035251, 38.69292214794663, 44.25879854861494, 23.081902178056566, 39.18805518137482, 41.676703672014206, 32.24106345081833, 17.82954627377335, 40.225687314290326, 23.99927680410368, 46.35423584805483, 45.02998536254883, 49.764132409510246, 27.22798217705092, 35.48912125308567, 28.634519175732258, 17.46102840575043, 24.24145332373876, 19.660409206954473, 46.99581756627617, 38.309242682466675, 40.31342349708809, 47.20874150000854, 46.320474900168925, 29.270446731783196, 49.101056992789125, 32.08650603279995, 41.44071530696042, 27.22214912564349, 41.902770963720535, 27.864717565245062, 42.15861653335925, 31.186353562984237, 49.35396205302669, 39.92098124533493, 44.10962700344452, 37.37996765623839, 39.037650478867775, 46.522252813493374, 17.252192397053694, 32.6794288474, 44.67497854649528, 17.905266266264807, 36.00668360662448, 27.990937728615645, 36.70574531827699, 44.59104679973061, 16.39504168551272, 14.536065330745673, 27.36113036095925, 14.824277928009987, 32.73672882759236, 44.842125850899336]
msemat=[93.32434783527381, 87.59191895427773, 90.92366192544473, 91.15674523645046, 90.26296023147744, 80.85918071603301, 86.7225460574156, 79.8411691656116, 77.56238687500411, 94.72126050264302, 84.0, 80.37383207538969, 85.08575646239228, 93.48723815027319, 78.6759162699412, 88.37774122341344, 86.52345257304692, 79.56249875767965, 82.14479967860714, 77.73223088259786, 92.83904927249225, 79.66514079847957, 69.04191173869226, 89.4335782397929, 94.1019195225598, 92.41502068825778, 91.67583572111283, 86.90187518387079, 82.79043129017094, 79.56479907006283, 89.13981068486666, 93.29793206038366, 79.8354565432116, 90.7976349935694, 81.56890429154242, 85.51776562108986, 90.28498058497082, 78.3427934985395, 91.18807376483025, 51.76673288603699, 82.46995748077511, 87.33452341032755, 94.65652355910282, 90.44794794721881, 89.9314191732796, 94.92629190300386, 66.2676010079483, 88.49056861604694, 87.05381685483607, 82.05476301922681, 92.62812435632485, 82.22075234825337, 89.84322719115352, 92.3413081692183, 77.26070928367812, 92.5291615392984, 88.41457273607205, 93.57259509846455, 80.33172997479483, 81.9489678590066, 60.64540968286502, 89.96931672360276, 78.25421523530845, 83.53782722574289, 92.23836710312781, 78.54390187273115, 78.41299511376121, 87.42899159501195, 87.61134198248396, 91.3134007403167, 71.45013225688372, 92.86496445380888, 74.94990899288098, 56.59599849696559, 76.05078200115811, 92.28386718703246, 90.89715838255962, 87.45265804824771, 60.76538769491113, 92.52789715233887, 85.33976524713579, 92.6157981826826, 94.14716029097468, 70.94429706689576, 82.75676923426167, 68.68336440106997, 79.8950283246231, 85.52804062013848, 80.55751657244092, 83.55451514908069, 93.79051088332196, 77.10472592594827, 87.4892061668489, 81.86371506921273, 94.28123732561608, 81.53331941092583, 89.02564361482345, 71.95673295949751, 79.82801842396931, 90.09525313512839]
'''
betamatABC=betamat
epsilonmatABC=epsilonmat
sigmamatABC=sigmamat
#mu1matABC=mu1mat
#mu2matABC=mu2mat
msematABC=msemat
EmatABC=Emat
alphamatABC=alphamat
omegamatABC=omegamat


for c in range(50):
    
    msemat=[]
    betamat=[]
    epsilonmat=[]
    sigmamat=[]
#    mu1mat=[]
#    mu2mat=[]
    Emat=[]
    alphamat=[]
    omegamat=[]
    
    mse_=np.median(msematABC)

    for b in range(iterations):

        mse=1000

        while mse>mse_:
            
            idays=[]
            ddays=[]
            cases=[]
            deaths=[]

            for i in range(len(df.index)):
                idays.append(df.loc[i, 'Day'])
                ddays.append(df.loc[i, 'Day'])
                cases.append(df.loc[i, 'Cumulative number of cases'])
                deaths.append(df.loc[i, 'Cumulative number of deaths']) 
            
            
            

            
                                    
            E=random.choice(EmatABC)
            t=idays[0]
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

 #           histmu1,binsmu1 = np.histogram(mu1matABC)
 #           bin_midpointsmu1 = binsmu1[:-1]+np.diff(binsmu1)/2
 #           cdfmu1=np.cumsum(histmu1)
 #           cdfmu1=cdfmu1 / cdfmu1[-1]
#            valuesmu1 = np.random.rand(1)
 #           value_binsmu1 = np.searchsorted(cdfmu1,valuesmu1)
 #           mu1 = bin_midpointsmu1[value_binsmu1] + random.uniform((binsmu1[0]-binsmu1[1])/2,(binsmu1[1]-binsmu1[0])/2)

 #           histmu2,binsmu2 = np.histogram(mu2matABC)
#            bin_midpointsmu2 = binsmu2[:-1]+np.diff(binsmu2)/2
#            cdfmu2=np.cumsum(histmu2)
 #           cdfmu2=cdfmu2 / cdfmu2[-1]
 #           valuesmu2 = np.random.rand(1)
#            value_binsmu2 = np.searchsorted(cdfmu2,valuesmu2)
 #           mu2 = bin_midpointsmu2[value_binsmu2] + random.uniform((binsmu2[0]-binsmu2[1])/2,(binsmu2[1]-binsmu2[0])/2)


            histalpha,binsalpha = np.histogram(alphamatABC)
            bin_midpointsalpha = binsalpha[:-1]+np.diff(binsalpha)/2
            cdfalpha=np.cumsum(histalpha)
            cdfalpha=cdfalpha / cdfalpha[-1]
            valuesalpha = np.random.rand(1)
            value_binsalpha = np.searchsorted(cdfalpha,valuesalpha)
            alpha = bin_midpointsalpha[value_binsalpha] + random.uniform((binsalpha[0]-binsalpha[1])/2,(binsalpha[1]-binsalpha[0])/2)

            histomega,binsomega = np.histogram(omegamatABC)
            bin_midpointsomega = binsomega[:-1]+np.diff(binsomega)/2
            cdfomega=np.cumsum(histomega)
            cdfomega=cdfomega / cdfomega[-1]
            valuesomega = np.random.rand(1)
            value_binsomega = np.searchsorted(cdfomega,valuesomega)
            omega = bin_midpointsomega[value_binsomega] + random.uniform((binsomega[0]-binsomega[1])/2,(binsomega[1]-binsomega[0])/2)





            SEIR(t, N, beta, epsilon, sigma, mu1, mu2, E, alpha, omega)

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

            

            caseserror=math.sqrt(np.sum((np.subtract(cases, cumifmat))**2))
            deathserror=math.sqrt(np.sum((np.subtract(deaths, cumdmat))**2))
            mse=caseserror# + deathserror
            
       
        betamat.append(beta)
        epsilonmat.append(epsilon)
        sigmamat.append(sigma)
#        mu1mat.append(mu1)
 #       mu2mat.append(mu2)
        Emat.append(E)
        msemat.append(mse)
        alphamat.append(alpha)
        omegamat.append(omega)
   

    
    betamatABC=betamat
    epsilonmatABC=epsilonmat
    sigmamatABC=sigmamat
#    mu1matABC=mu1mat
#    mu2matABC=mu2mat
    msematABC=msemat
    EmatABC=Emat
    alphamatABC=alphamat
    omegamatABC=omegamat

    print('betamat',c+2,'=',betamat)
    print('epsilonmat',c+2,'=',epsilonmat)
    print('sigmamat',c+2,'=',sigmamat)
#    print('mu1mat',c+2,'=',mu1mat)
#    print('mu2mat',c+2,'=',mu2mat)
    print('msemat',c+2,'=',msemat)
    print('Emat',c+2,'=',Emat)
    print('alphamat',c+2,'=',alphamat)
    print('omegamat',c+2,'=',omegamat)   

  

# HISTOGRAM #

#plt.hist(sigmamat, density=1)
#plt.show()
