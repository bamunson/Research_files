'''
AUTHOR: BRAD MUNSON
This script takes a data file in the format shown in 'data.columns' from an 
AMR mesh grid and spherically averages the quantities weighted by volume or mass.
Angular momentum is calculated cylindrically as that is how stars rotate.
'''
#PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import helmholtz
from scipy.interpolate import interp1d
import rcbtools as rcb

t_start = time.time()

#Inputs
rmax = 3.4478e10
resolution = 1500 # higher number gives more bins (empty bins will automatically be thrown away)
binning = 'log' # Options are log or linear
savefile = 'fullDT.dat' # File name to save averaged data into
net = 'mesa_75' # Options are mesa_75, cno_extras_to_mg26, sagb, pp_cno_extras, and hot_cno
Read_data = True
lowZ = True
custom_abund = True
initial_M = 0.85 #Changing this mass will rescale the entire system assuming mass radius relationship
                 #for non-relativistic WD and the gravitational constant in octotiger is 1.

#Constants and Scaling factors
Msun = 1.99e+33
Rsun = 6.9634e10
G = 6.67e-8
mcon = initial_M / 0.88
lcon = (initial_M*0.6)**(-1/3)*637.3e6/0.2 / 3.271e9
tcon = ((lcon*3.271e9)**3/G/(mcon*1.988e33))**0.5 / 16.22

#READING IN THE DATA (THE MESSY BUT FAST WAY) AND RESCALE EVOLVED VARIABLES
if Read_data:
    print('Reading Data')
    t0 = time.time()
    data = pd.read_csv('format_data.dat', sep=" ", usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),header=None)
    data.columns = ['ind','xl','xr','yl','yr','zl','zr','vx','vy','vz','sx','sy','sz','dm','rho','eint','tau','he4','c12','pot','phi']
    ind = np.array(data.ind)
    xl = np.array(data.xl) * lcon
    xr = np.array(data.xr) * lcon
    yl = np.array(data.yl) * lcon
    yr = np.array(data.yr) * lcon
    zl = np.array(data.zl) * lcon
    zr = np.array(data.zr) * lcon
    vx = np.array(data.vx) * (lcon/tcon)
    vy = np.array(data.vy) * (lcon/tcon)
    vz = np.array(data.vz) * (lcon/tcon)
    sx = np.array(data.sx) * (mcon * lcon / tcon) / (lcon**3)
    sy = np.array(data.sy) * (mcon * lcon / tcon) / (lcon**3)
    sz = np.array(data.sz) * (mcon * lcon / tcon) / (lcon**3)
    dm = np.array(data.dm) * (mcon)
    rho = np.array(data.rho) * (mcon / lcon**3)
    eint = np.array(data.eint) * (lcon**2 / tcon **2)
    tau = np.array(data.tau) * ((mcon * lcon**2 / tcon**2) / (lcon**3))**(3/5)
    secondary = np.array(data.he4)
    primary = 2*np.array(data.c12)
    pot = np.array(data.pot) * mcon * lcon**2 / tcon**2 / lcon**3
    phi = np.array(data.phi) * lcon**2 / tcon**2 
    t1 = time.time()
    print(t1-t0,'s')

#initialize composition (may need to be user edited for different networks)
if net == 'cno_extras_to_mg26':
#h1, he3, he4, c12, c13, n13, n14, n15, o14, o15, o16, o17, o18, f17, f18, 
#f19, ne18, ne19, ne20, ne22, mg22, mg24, mg26
    s_frac = np.array([3.39741198e-05, 8.36724050e-09, 9.90217413e-01, 4.21733145e-05,\
                 1.17422489e-05, 2.53516187e-99, 6.14504526e-03, 2.56404898e-07,\
                 1.79050106e-99, 2.66052369e-99, 4.40823552e-04, 1.34786032e-05,\
                 4.85287274e-09, 2.81955118e-99, 1.72854273e-99, 1.52064930e-09,\
                 1.33512384e-99, 6.63947393e-99, 9.66842806e-04, 7.85507701e-05,\
                 4.77073437e-06, 3.47386717e-04, 1.69752724e-03])
    p_frac = np.array([7.41783621e-40, 3.18677240e-23, 3.56752143e-27, 3.14430876e-01,\
                 4.08533252e-05, 6.83390249e-32, 3.81142785e-16, 1.98064420e-15,\
                 4.71726175e-62, 5.46414571e-46, 6.59744204e-01, 2.34187328e-06,\
                 3.14365001e-17, 2.37043739e-34, 9.64837431e-38, 5.66996967e-07,\
                 1.93856270e-68, 1.41073377e-69, 2.02105025e-03, 1.93262169e-02,\
                 1.76562028e-05, 3.95042972e-03, 4.65804441e-04])
    a_ele = np.array([1,3,4,12,13,13,14,15,14,15,16,17,18,17,18,19,18,19,20,22,22,24,26])
    z_ele = np.array([1,2,2,6,6,7,7,7,8,8,8,8,8,9,9,9,10,10,10,10,12,12,12])
if net == 'hot_cno':
#h1, he3, he4, c12, c13, n13, n14, n15, o15, o16, o17, o18, f17, f18, ne20, mg24
    s_frac = np.array([3.68800081e-05, 9.89785652e-09, 9.90214629e-01, 4.16953834e-05,\
                 1.15761522e-05, 2.09320955e-99, 6.14448604e-03, 2.56457861e-07,\
                 2.41132164e-99, 4.42223805e-04, 1.33459269e-05, 4.62728469e-09,\
                 2.06937696e-99, 1.74207527e-99, 1.04973024e-03, 2.04516220e-03])
    p_frac = np.array([4.45025974e-23, 5.05481693e-21, 1.12923669e-04, 4.08799574e-01,\
                 4.10209615e-05, 6.33692346e-33, 5.24885602e-15, 4.48514334e-07,\
                 5.60452138e-51, 5.68790196e-01, 2.34438577e-06, 1.60596947e-02,\
                 2.11352335e-38, 3.96168417e-42, 2.10371817e-03, 4.09007962e-03])
    a_ele = np.array([1,3,4,12,13,13,14,15,15,16,17,18,17,18,20,24])
    z_ele = np.array([1,2,2,6,6,7,7,7,8,8,8,8,9,9,10,12])
if net == 'sagb':
#neut, h1, h2, he3, he4, li7, be7, b8, c12, c13, n13, n14, n15, o16, o17, o18, f19
#ne20, ne21, ne22, na21, na22, na23, mg24, mg25, mg26, al25, al26, al27 
    p_frac = np.array([3.22927438e-06, 1.26312693e-28, 1.08069619e-21, 2.50536510e-30,\
          1.35506390e-03, 1.66150721e-21, 7.96028026e-39, 1.94904612e-67,\
          3.44539650e-01, 6.77383695e-10, 4.78300799e-20, 6.99339211e-15,\
          4.49202857e-07, 6.28375177e-01, 2.05212969e-06, 1.60217360e-15,\
          4.55890709e-10, 1.96486420e-03, 2.25678384e-05, 1.95192006e-02,\
          8.59846537e-27, 5.06545225e-19, 1.67615758e-04, 3.84722799e-03,\
          3.09934760e-05, 1.13871042e-04, 4.05008905e-23, 1.26774153e-14,\
          5.80356823e-05])
    s_frac = np.array([7.29659838e-16, 3.78634108e-05, 3.40121347e-21, 8.77293126e-09,\
           9.90209914e-01, 1.29464794e-21, 1.72424316e-25, 1.08605454e-61,\
           4.17849745e-05, 1.16104764e-05, 1.62043502e-99, 6.14556422e-03,\
           2.56509721e-07, 4.40997255e-04, 1.33806415e-05, 4.68189959e-09,\
           1.44506513e-09, 9.64519499e-04, 7.44095639e-06, 1.96685649e-10,\
           2.11782812e-99, 2.80982246e-99, 1.02297754e-04, 3.11672003e-04,\
           3.54408470e-05, 5.28582133e-05, 1.98234966e-99, 8.96526182e-99,\
           1.62438398e-03])
    a_ele = np.array([1,1,2,3,4,7,7,8,12,13,13,14,15,16,17,18,19,20,21,22,21,22,23,24,\
                      25,26,25,26,27])
    z_ele = np.array([1,1,1,2,2,3,4,5,6,6,7,7,7,8,8,8,9,10,10,10,11,11,11,12,12,12,13,13,13])
    
if net == 'pp_cno_extras':
    #pp_cno_extras_o18_ne22 - h1 h2 he3 he4 li7 be7 be8 c12 c13 n13 n14 n15 o14 o15
    # o16 o17 o18 f17 f18 f19 ne18 ne19 ne20 ne22 mg22 mg24
    #p_frac = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    #s_frac = np.array([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    p_frac = np.array([5.64510787e-020, 4.41913541e-023, 2.06092416e-022, 2.35643154e-005,\
                         1.00636367e-020, 2.91167689e-036, 6.12442903e-063, 3.80540649e-001,\
                         4.15204027e-005, 2.08511786e-032, 1.70400427e-013, 3.33190895e-013,\
                         8.48574786e-066, 1.57558918e-051, 5.93573896e-001, 2.19043375e-006,\
                         3.37465859e-016, 1.03107806e-038, 2.67307126e-042, 5.67460918e-007,\
                         7.09083858e-101, 7.09082536e-101, 2.04286855e-003, 1.97556815e-002,\
                         1.76958409e-005, 4.00136646e-003])
    s_frac = np.array([3.255248423831610109e-05,4.359178080847557632e-21,7.689384052958702031e-09,9.902188953422885742e-01,\
                       1.470685008203123568e-21,6.432712691851882787e-30,8.578096254781798028e-62,4.230953176020302742e-05,\
                       1.179137318397046186e-05,1.360970253922478255e-99,6.144682690503912130e-03,2.563978839626313880e-07,\
                       1.889211009243815955e-99,1.571274243215122938e-99,4.415123658430637667e-04,1.291303881528981884e-05,\
                       4.693516014079674582e-09,1.426187026410107488e-99,1.905055902346840366e-99,1.490598062038396206e-09,\
                       1.467728246546097663e-99,4.557173828338925366e-99,9.668350934679111270e-04,7.855077010969172696e-05,\
                       4.779216283599998124e-06,2.044907821299611107e-03])
    a_ele = np.array([1,2,3,4,7,7,8,12,13,13,14,15,14,15,16,17,18,17,18,19,18,19,20,22,22,24])
    z_ele = np.array([1,1,2,2,3,4,4,6,6,7,7,7,8,8,8,8,8,9,9,9,10,10,10,10,12,12])

#MESA_75 COMPOSITION
if net == 'mesa_75':
    #neut, h1, h2, he3, he4, li7, be7, be9, be10, b8, c12, c13, n13, n14, n15, o14, o15, o16, o17, o18, 
    #f17, f18, f19, ne18, ne19, ne20, ne21, ne22, na21, na22, na23, na24, mg23, mg24, mg25, mg26, al25, 
    #al26, al27, si27, si28, p30, p31, s31, s32, cl35, ar35, ar36, ar37, ar38, k39, ca39, ca40, ca41, 
    #ca42, sc43, ti44, ti45, ti46, v47, cr48, cr49, cr50, mn51, fe52, fe53, fe54, fe56, co55, co56, ni56, 
    #ni57, ni58, cu59, zn60
    
    #p_frac = np.zeros((75,))
    #p_frac[[10,17]] = 0.5
    
    if lowZ:
        p_frac = np.array([5.04408475e-034, 2.43812141e-024, 1.17336674e-022, 2.04636505e-018,\
                             9.87551713e-010, 2.08602007e-022, 5.92914190e-037, 9.51221742e-026,\
                             4.73217796e-022, 1.06202872e-067, 3.71492775e-001, 4.50970669e-008,\
                             7.10504917e-033, 2.43129693e-013, 1.33219194e-017, 1.99062266e-066,\
                             1.42839225e-048, 6.25804100e-001, 7.91420397e-007, 5.96318933e-014,\
                             1.31329814e-037, 6.92562707e-043, 6.09860685e-007, 3.94545734e-100,\
                             5.14382873e-100, 2.33835688e-004, 1.52256675e-005, 1.94825878e-003,\
                             1.44868858e-099, 1.70836718e-099, 4.08316135e-005, 3.06133639e-100,\
                             3.06133639e-100, 6.47947077e-005, 1.60833387e-005, 5.42981033e-005,\
                             3.06133639e-100, 3.06133639e-100, 8.95289557e-006, 3.06133639e-100,\
                             8.82648408e-005, 3.74446048e-044, 9.76769217e-007, 3.06133639e-100,\
                             4.39835130e-005, 5.86611351e-007, 3.06133639e-100, 6.16959910e-006,\
                             9.58283468e-007, 2.22509504e-006, 4.57899255e-007, 3.06133639e-100,\
                             6.96425032e-006, 9.09797748e-008, 1.10580487e-007, 3.46870840e-042,\
                             3.29602191e-046, 1.75981208e-030, 3.62866221e-007, 3.50767355e-019,\
                             9.96100400e-035, 1.09133409e-035, 2.21623898e-006, 1.68815556e-021,\
                             1.44191142e-034, 1.02088642e-033, 8.95967900e-006, 1.45516393e-004,\
                             5.45739950e-019, 3.06133639e-100, 3.06133639e-100, 2.66074017e-007,\
                             9.41920118e-006, 6.52652493e-020, 1.86799244e-006])
        
        s_frac = np.array([0.00000000e+00, 1.15082407e-03, 2.23967994e-19, 5.57061958e-06,\
                            9.96893992e-01, 4.59686678e-21, 1.44186036e-25, 8.01201102e-36,\
                            2.90374670e-50, 1.85735065e-59, 7.77822866e-06, 2.15551075e-06,\
                            1.05959548e-99, 1.24163007e-03, 5.27473951e-08, 5.02040159e-99,\
                            1.63186964e-99, 7.02807612e-05, 7.55161868e-06, 2.10612270e-09,\
                            1.22367879e-99, 1.41521605e-99, 5.43286795e-10, 5.43198070e-99,\
                            7.99112291e-99, 1.88178319e-04, 6.69480581e-06, 2.15176121e-11,\
                            1.23813795e-99, 1.48774641e-99, 2.01906814e-05, 1.07597199e-99,\
                            7.59967607e-99, 6.23346004e-05, 1.59065275e-06, 1.61316604e-05,\
                            1.16211845e-99, 9.54301391e-99, 7.39419199e-06, 1.90935520e-99,\
                            8.82421213e-05, 7.75545592e-50, 9.73165727e-07, 1.03059548e-98,\
                            4.39768626e-05, 5.84582739e-07, 1.00027754e-99, 7.28529153e-06,\
                            6.42986898e-18, 1.39807312e-06, 4.56405504e-07, 1.00019136e-99,\
                            7.71021210e-06, 2.66915346e-18, 5.40294626e-08, 1.24865556e-20,\
                            1.06751495e-34, 3.83744735e-46, 3.60768996e-07, 1.00367041e-22,\
                            3.03016332e-44, 1.50434813e-55, 2.20375732e-06, 9.91589974e-25,\
                            2.09220755e-44, 1.00411057e-55, 8.84443068e-06, 1.43974548e-04,\
                            9.72083483e-27, 2.38860902e-99, 1.95888946e-98, 1.81504613e-18,\
                            9.03196542e-06, 3.10665922e-29, 2.55068497e-06])
    else:
        p_frac = np.array([6.46219402e-33, 1.32378627e-28, 9.26957469e-49, 9.21809015e-38,\
                   7.60306034e-18, 1.19772637e-19, 1.63428338e-48, 1.34148519e-45,\
                   1.01004456e-32, 3.82204258e-78, 3.29924342e-01, 7.49860539e-10,\
                   4.88602700e-20, 3.34732635e-15, 1.28424941e-12, 5.29413187e-40,\
                   6.29330345e-34, 6.44162051e-01, 1.73431281e-06, 1.14171350e-13,\
                   2.56352798e-23, 1.46379376e-26, 1.52834941e-05, 2.79488530e-47,\
                   3.56897954e-48, 1.51288533e-03, 3.62992039e-05, 1.91213981e-02,\
                   6.53127256e-27, 6.10003022e-19, 1.16278889e-04, 1.70508924e-20,\
                   6.91122112e-36, 2.70885824e-03, 1.57405958e-04, 4.29288702e-04,\
                   2.94182908e-23, 6.67227261e-21, 4.17772968e-05, 4.66120078e-32,\
                   4.61727837e-04, 3.26693071e-22, 5.76620949e-06, 4.24023782e-44,\
                   2.79864586e-04, 1.79029630e-06, 5.15324028e-74, 4.12610620e-05,\
                   1.08468140e-05, 1.93898494e-05, 2.45529385e-06, 2.07379871e-98,\
                   3.56487333e-05, 5.00716069e-07, 8.09311190e-07, 1.68926702e-18,\
                   1.40201809e-23, 3.81608099e-24, 1.57959025e-07, 1.46789032e-20,\
                   7.58752047e-26, 1.83383983e-25, 5.24927408e-07, 2.22619537e-21,\
                   2.54602534e-36, 1.42482247e-25, 5.04146681e-05, 8.26279927e-04,\
                   1.53151895e-20, 2.32230876e-33, 1.92519854e-44, 2.56524409e-25,\
                   3.49581587e-05, 1.15919678e-21, 3.09574324e-25])
        
        s_frac = np.array([0.0,2.86710471551e-05,3.82197054578e-21,8.16753159779e-09,0.990219210452,2.1594322892e-08,\
               1.53793680631e-25,1.19003007763e-38,3.05189849183e-49,2.8671427823e-62,4.22212633534e-05,1.18049313691e-05,\
               1.01935258172e-99,0.00614595325669,2.56815877751e-07,5.34483690383e-99,1.42526725789e-99,0.000439521446933,\
               1.36111149308e-05,4.94783465378e-09,1.30718019633e-99,2.29713572637e-99,1.47522681148e-09,6.23425721091e-99,\
               8.98951880295e-99,0.000964495243845,7.47378166573e-06,2.63078381171e-10,1.0743694693e-99,3.40858397792e-99,\
               0.00010228963807,1.29949514116e-99,7.68576912745e-99,0.000311664895495,3.54081801053e-05,5.28913640999e-05,\
               1.03736149564e-99,7.42046143765e-99,3.61588000052e-05,2.00193254054e-99,0.000441209704678,2.27329736304e-46,\
               4.86582828843e-06,1.16132737994e-98,0.000219884159311,2.92291440828e-06,1.00032505257e-99,3.64264594771e-05,\
               2.85448475282e-15,6.9903659499e-06,2.2820276803e-06,1.00028142519e-99,3.85510428692e-05,1.45222587127e-15,\
               2.70147189261e-07,3.01941491494e-21,1.34845290138e-36,6.82230289797e-46,1.80384484873e-06,2.41769179894e-23,\
               3.39446268753e-45,2.31860271965e-54,1.10187859759e-05,2.67726050364e-25,4.2840117731e-46,2.74065627507e-55,\
               4.42221448011e-05,0.000719872600958,3.24874607074e-27,1.73997743444e-99,1.95947409707e-98,1.15525605659e-15,\
               4.51598183689e-05,1.33823867403e-29,1.27538118248e-05])
    
    a_ele = np.array([1,1,2,3,4,7,7,9,10,8,12,13,13,14,15,14,15,16,17,18,17,18,19,18,19,20,21,22,21,22,23,24,\
                      23,24,25,26,25,26,27,27,28,30,31,31,32,35,35,36,37,38,39,39,40,41,42,43,44,45,46,47,\
                      48,49,50,51,52,53,54,56,55,56,56,57,58,59,60])
    z_ele = np.array([1,1,1,2,2,3,4,4,4,5,6,6,7,7,7,8,8,8,8,8,9,9,9,10,10,10,10,10,11,11,11,11,\
                      12,12,12,12,13,13,13,14,14,15,15,16,16,17,18,18,18,18,19,20,20,20,20,21,22,22,\
                      22,23,24,24,24,25,26,26,26,26,27,27,28,28,28,29,30])

if net == 'rcb':
    #neut, h1, h2, he3, he4, li7, be7, be9, be10, b8, b11, c11, c12, c13, c14, 
    #n13, n14, n15, o14, o15, o16, o17, o18, f17, f18, f19, ne18, ne19, ne20, 
    #ne21, ne22, na21, na22, na23, na24, mg23, mg24, mg25, mg26, fe56
    
    if lowZ:
        p_frac = np.array([2.82692451e-62, 2.04467778e-44, 9.29678035e-30, 1.93259896e-25,\
                           6.88264494e-27, 6.00314222e-29, 5.79106488e-56, 4.95162833e-46,\
                           1.74907403e-34, 1.11296450e-99, 5.94750468e-26, 1.44332219e-99,\
                           3.66749488e-01, 1.49738371e-07, 1.73244194e-12, 6.66141482e-48,\
                           5.08782752e-10, 3.47228471e-14, 6.07156962e-96, 3.99266255e-63,\
                           6.30524187e-01, 9.03626630e-07, 6.84641165e-11, 4.74432258e-52,\
                           6.94231920e-57, 3.43287967e-07, 4.48264763e-99, 5.44411686e-99,\
                           2.70261656e-04, 2.11832461e-05, 1.92269083e-03, 1.84659525e-99,\
                           1.62761150e-99, 3.83006942e-05, 2.36014014e-99, 4.96567758e-99,\
                           6.36770665e-05, 2.88584415e-05, 5.50933633e-05, 3.24861552e-04])
        
        s_frac = np.array([0.00000000e+00, 4.61632601e-04, 9.06072528e-20, 9.66959640e-07,\
                           9.98537207e-01, 3.17124560e-21, 4.68735028e-25, 5.78374444e-35,\
                           1.15186760e-46, 2.05192243e-62, 1.22891239e-30, 1.20815582e-99,\
                           5.03479307e-06, 1.40909177e-06, 2.30548120e-69, 1.02231750e-99,\
                           6.43017596e-04, 2.65543033e-08, 4.81914258e-99, 1.50045829e-99,\
                           3.16403393e-05, 4.72526835e-07, 2.06321905e-10, 1.03806022e-99,\
                           2.23819593e-99, 4.51455697e-11, 1.23325127e-99, 7.46956818e-99,\
                           9.35368881e-05, 6.83568770e-06, 4.73530728e-11, 1.22739875e-99,\
                           1.59429113e-99, 1.02542942e-05, 1.52627404e-99, 1.29546104e-99,\
                           3.20225000e-05, 4.22414035e-06, 4.83266516e-06, 1.66885714e-04])
    else:
        p_frac = np.array([1.29549904e-65, 4.07374654e-51, 2.45845476e-28, 4.94617834e-29,\
                           3.82578789e-28, 4.37499227e-31, 4.39438322e-61, 1.50503759e-45,\
                           4.66827543e-35, 1.04011221e-99, 6.57877953e-28, 1.70766611e-99,\
                           3.61078322e-01, 1.53735354e-09, 1.21899288e-12, 6.03191962e-52,\
                           1.64058836e-09, 7.35942138e-13, 2.34746709e-99, 4.63826774e-66,\
                           6.12932136e-01, 2.50076776e-06, 4.74181356e-10, 1.44336903e-55,\
                           5.61406910e-59, 5.85092721e-06, 1.77994253e-99, 2.64538565e-99,\
                           2.11377645e-03, 5.66675270e-05, 1.91768832e-02, 3.05965748e-99,\
                           3.06077601e-99, 1.89902007e-04, 1.94639996e-99, 3.03815988e-99,\
                           6.21455314e-04, 1.66468462e-04, 4.07417978e-04, 3.24861550e-03])
        
        s_frac = np.array([0.00000000e+00, 2.71175042e-04, 3.65541648e-20, 4.78557112e-07,\
                           9.89728035e-01, 2.04592685e-21, 1.31143826e-25, 4.20876365e-35,\
                           5.54314662e-47, 7.64872785e-64, 5.74272922e-30, 1.07870957e-99,\
                           4.62549920e-05, 1.29292506e-05, 2.20981998e-68, 1.89284913e-99,\
                           6.29019603e-03, 2.60544027e-07, 5.51299956e-99, 1.04700610e-99,\
                           4.54015093e-04, 1.95626436e-05, 6.42069914e-09, 1.50489478e-99,\
                           1.06065169e-99, 1.52044438e-09, 1.09717732e-99, 3.87048698e-99,\
                           9.89047035e-04, 7.60173354e-06, 3.20734375e-11, 1.24230826e-99,\
                           1.74229309e-99, 1.04896724e-04, 1.07047634e-99, 1.43211407e-99,\
                           3.19581699e-04, 4.21581505e-05, 4.82314053e-05, 1.66556801e-03])
            
    a_ele = np.array([1,1,2,3,4,7,7,9,10,8,11,11,12,13,14,\
                      13,14,15,14,15,16,17,18,17,18,19,18,19,20,\
                      21,22,21,22,23,24,23,24,25,26,56])
        
    z_ele = np.array([1,1,1,2,2,3,4,4,4,5,5,6,6,6,6,\
                      7,7,7,8,8,8,8,8,9,9,9,10,10,10,\
                      10,10,11,11,11,11,12,12,12,12,26])

#First let's compute box-center values
print('Transforming to CoM and face center values')
t0 = time.time()
xc = (xl+xr)/2
yc = (yl+yr)/2
zc = (zl+zr)/2
dV = abs((xl-xr)*(yl-yr)*(zl-zr))
    
#Find CoM positional values (based on maximum density)
center = np.argmax(rho)
x0 = xc - xc[center]
y0 = yc - yc[center]
z0 = zc - zc[center]
r0 = np.sqrt(x0**2+y0**2+z0**2)
r0[center] = np.cbrt(3 * dV[center] / np.pi / 4)
R0 = np.sqrt(x0**2 + y0**2)
R0[np.where(R0 == 0)] = r0[center]


#Computing angular momentum (cylindrical) and kinetic energy
#ek uses momentum because it is in the inertial frame
j = abs((x0*sy-y0*sx)/rho)
ek = 0.5*(sx**2+sy**2+sz**2)/rho**2
if rmax<0:
    rmax = r0.max()/np.sqrt(3)

t1 = time.time()
print(t1-t0,'s')

#Define radial grid that will take care of the binning
if binning == 'linear':
    rr = np.linspace(0,rmax,resolution)
    dr = rr[1]-rr[0]
    print('Linear Binning')
    
if binning == 'log':
    rr = np.logspace(0,np.log10(rmax),resolution)
    dr = np.log10(rr[1])-np.log10(rr[0])
    logrmax = np.log10(rmax)
    Rr = np.logspace(0,np.log10(rmax),int(resolution/10))
    dR = np.log10(Rr[1])-np.log10(Rr[0])
    print('Log10 Binning')
    
print('Beginning the Spherical Averaging')
t0 = time.time()

rho_bar = np.zeros(np.size(rr))
j_bar = np.zeros(np.size(Rr))
j_sph = np.zeros(np.size(rr))
m_bar = np.zeros(np.size(rr))
V_bar = np.zeros(np.size(rr))
V_cyl = np.zeros(np.size(Rr))
tau_bar = np.zeros(np.size(rr))
q = np.zeros(np.size(rr))
primary_bar = np.zeros(np.size(rr))
secondary_bar = np.zeros(np.size(rr))
ek_bar = np.zeros(np.size(rr))
m_cyl = np.zeros(np.size(Rr))


#Start binning into final radial grid
if binning == 'log':
    for i,r in enumerate(np.log10(r0+1)):
        if r <= logrmax:
            n = int(r/dr+1)
            m = int(np.log10(R0[i])/dR + 1)
            rho_bar[n] += rho[i]*dV[i]
            j_bar[m] += j[i]*dm[i]
            j_sph[n] += j[i]*dm[i]
            m_bar[n] += dm[i]
            m_cyl[m] += dm[i]
            V_bar[n] += dV[i]
            V_cyl[m] += dV[i]
            secondary_bar[n] += secondary[i]*dm[i]
            primary_bar[n] += primary[i]*dm[i]
            tau_bar[n] += tau[i]*dV[i]
            ek_bar[n] += ek[i]*dV[i]

if binning == 'linear':
    for i,r in enumerate(r0):
        if r <= rmax:
            n = int(r/dr+0.5)
            rho_bar[n] += rho[i]*dV[i]
            j_bar[n] += j[i]*dV[i]
            m_bar[n] += dm[i]
            V_bar[n] += dV[i]
            secondary_bar[n] += secondary[i]*dm[i]
            primary_bar[n] += primary[i]*dm[i]
            tau_bar[n] += tau[i]*dV[i]
            ek_bar[n] += ek[i]*dV[i]
            
t1 = time.time()
print(t1-t0,'s')

#Find any bad values (empty cells) and remove them
print('Removing empty cells from data')
t0 = time.time()
i_bad = np.where(V_bar==0)
j_bad = np.where(j_bar==0)

rr = np.delete(rr,i_bad)
rho_bar = np.delete(rho_bar,i_bad)
Rr = np.delete(Rr,j_bad)
j_bar = np.delete(j_bar,j_bad)
j_sph = np.delete(j_sph,i_bad)
m_bar = np.delete(m_bar,i_bad)
V_bar = np.delete(V_bar,i_bad)
V_cyl = np.delete(V_cyl,j_bad)
m_cyl = np.delete(m_cyl,j_bad)
tau_bar = np.delete(tau_bar,i_bad)
q = np.delete(q,i_bad)
secondary_bar = np.delete(secondary_bar,i_bad)
primary_bar = np.delete(primary_bar,i_bad)
ek_bar = np.delete(ek_bar,i_bad)


j_sph = j_sph/m_bar
tau_bar = tau_bar/V_bar
rho_bar = rho_bar/V_bar
eint_bar = (tau_bar**(5/3))/rho_bar
j_bar = j_bar/m_cyl
secondary_bar = secondary_bar/m_bar
primary_bar = primary_bar/m_bar
ek_bar = ek_bar/V_bar
mtot = np.sum(m_bar)
mtot_cyl = np.sum(m_cyl)
t1 = time.time()
print(t1-t0,'s')

#Calculating q coordinate (normalized mass exterior to shell)
#Also finding the potential contribution from mass outside shell
uint = np.zeros(np.shape(m_bar))
for i,m in enumerate(m_bar):
    q[i] = 1-(np.sum(m_bar[0:i+1])/mtot)
    uint[i] = np.trapz(rr[i:]*rho_bar[i:],rr[i:])
q_cyl = np.array([1-np.sum(m_cyl[0:i+1])/mtot_cyl for i in range(len(m_cyl))])
mr1 = mtot*(1-q)

print('Total mass [solar]:',mtot/Msun)
print('Number of bad cells:',resolution-len(rr))

#COMPUTE ABUNDANCE PROFILE FROM TRACER
NELE = len(p_frac)
abund = primary_bar.reshape(len(rr),1) @ p_frac.reshape(NELE,1).T + secondary_bar.reshape(len(rr),1) @ s_frac.reshape(NELE,1).T


#Determine abar, zbar based on OctoTiger composition ONLY
abar_octo = np.zeros(np.shape(rr))
zbar_octo = np.zeros(np.shape(rr))
for i in range(len(a_ele)):
    abar_octo += abund[:,i]/a_ele[i]
    zbar_octo += abund[:,i]/z_ele[i]
abar_octo = 1/abar_octo
zbar_octo = 1/zbar_octo

#Use Timmes Helmholtz EOS to reconstruct OctoTiger profile consistent with the MESA EoS, at least for HeWD (non-degenerate) component
print('Using helmeos to compute temperature, pressure, and entropy')         
h = helmholtz.helmeos_DE(dens=rho_bar, ener=eint_bar*rho_bar, abar = 4, zbar = 2, tguess = 1e7)

temp_helm = h.temp
s_helm = h.stot
p_helm = h.ptot
print('Inserting isothermal core')
#Insert the isothermal core
if custom_abund:
    i_iso = np.argmin(abs(mr1/Msun - initial_M/1.6))
    temp_helm[:i_iso+1] = 1e7
    i_peak = np.argmax(temp_helm)
    fit = np.delete(temp_helm[:i_peak],np.where(temp_helm[:i_peak]<1e7))
    q_fit = np.delete(q[:i_peak],np.where(temp_helm[:i_peak]<1e7))
    f = interp1d(q_fit,fit,kind='linear')
    temp_helm[:i_peak] = f(q[:i_peak])
    i_trans = i_iso #Put composition transition at end of isothermal core
    #i_trans = np.argmax(temp_helm)
    abund[:i_trans+1] = p_frac
    abund[i_trans+1:] = s_frac
else:  
    for i,T in enumerate(temp_helm):
        if T <= 1e7 and rho_bar[i] > 1e4:
            temp_helm[i] = 1e7
    
        
#Save data file(s)

header = str(len(rr))
DT_data = np.vstack((q,rho_bar,temp_helm)).T
am_data = np.vstack((q_cyl,j_bar)).T
abund_data = np.vstack((q,abund.T)).T
abund_data = abund_data[::-1]
DT_data = DT_data[::-1]
am_data = am_data[::-1]
np.savetxt('eosDT.dat',DT_data,header=header,comments='')
header = str(len(Rr))
np.savetxt('am.dat',am_data,header=header,comments='')
header = str(len(rr))+' '+str(NELE)
np.savetxt('abund.dat',abund_data,header=header,comments='')
t_done = time.time()
print('Done! Total time:', t_done-t_start,'s')


#Plots to check the averaged values MESA will read
plt.figure(100)
plt.subplot(221)
plt.semilogy(q,eint_bar)
plt.ylabel('Eint')
plt.subplot(222)
plt.semilogy(q_cyl,j_bar)
plt.ylabel('J')
plt.subplot(223)
plt.semilogy(q,rho_bar)
plt.ylabel('rho')
plt.xlabel('q')
plt.subplot(224)
plt.semilogy(q,temp_helm)
plt.ylabel('T')
plt.xlabel('q')
plt.legend()

plt.figure(200)
plt.loglog(rho_bar,temp_helm)
plt.xlabel('log(Rho)')
plt.ylabel('log(T)')
plt.show()


u = -G*mr1/rr-4*np.pi*G*uint

etot = ek_bar+u+eint_bar
plt.figure(300)
plt.subplot(211)
plt.plot(rr,mr1/Msun)
plt.ylabel('Interior Mass')
plt.subplot(212)
plt.plot(rr,etot,'.')
#plt.yscale('symlog')
plt.axhline(0,color='k')
#plt.semilogx(rr,abs(u))
plt.ylabel('Total Energy')
plt.xlabel('Radius')

print('Resolution:',resolution)
print('q_abund:',q[np.argmin(abs(primary_bar-secondary_bar))])
print('q_mass:',1-(0.6*initial_M)/(mtot/Msun))
print('r_bound:',rr[np.argmin(abs(etot[10:]))+10])
print('m_bound:',mr1[np.argmin(abs(etot[10:]))+10]/Msun)



