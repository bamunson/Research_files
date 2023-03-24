#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:54:27 2021

@author: bradmunson
"""
import Octo2Yt as ot
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import rcbtools as r
import helmholtz

#USER PARAMETERS
filename = 'X.5100.silo.yt.npz' #Name of Octo-Tiger AMR file (compressed or not)
rmax = -1
resolution = 1000
read_data = True

G = 6.67e-8
Msun = 1.99e33
mu_e = 2
c = 2.99792458e10
m_p = 1.6726e-24
m_e = 9.1094e-28
h = 6.626e-27
eps_1 = 0.001
kB = 1.3807e-16
mp = 1.6726e-24
Rsun = 6.957e10
AU2cm = 1.496e+13

A = np.pi * m_e**4 * c**5 / (3*h**3) #Energy/Volume
B = 8 * np.pi * m_p * mu_e * (m_e * c / h)**3 / 3

if read_data:
    print('Reading Data...')
    
    #Read in file data
    if 'yt.npz' in filename:
        ds = ot.loadFromNPZ(filename) #Filename if compressed
    else:
        ds = ot.octo2yt_amr(filename, nspecies=5, gather_outflows=False, savefile=True, copy_path='./') #Filename if not compressed
    
    print('Done reading, processing data...')
    
    #Format data to numpy-like structure
    data = ds.all_data() 
    mtot = np.sum(data['stream','density']*data['index','cell_volume']*ds.length_unit**3)
    print('Total mass: ',mtot)
    
    #Compute relavent data and store in dictionary of numpy arrays
    d = {}
    i_center = np.argmax(data['stream','density'])
    d['x'] = np.array((data['index','x'] - data['index','x'][i_center])*ds.length_unit)
    d['y'] = np.array((data['index','y'] - data['index','y'][i_center])*ds.length_unit)
    d['z'] = np.array((data['index','z'] - data['index','z'][i_center])*ds.length_unit)
    d['dV'] =  np.array(data['index','cell_volume']*ds.length_unit**3)
    d['r'] = np.sqrt(d['x']**2+d['y']**2+d['z']**2)
    d['R'] = np.sqrt(d['x']**2+d['y']**2)
    d['tau'] = np.array(data['stream','tau'])
    d['rho'] = np.array(data['stream','density'])
    d['sx'] = np.array(data['stream','sx']/d['rho'] - data['stream','sx'][i_center]/d['rho'][i_center])*d['rho']
    d['sy'] = np.array(data['stream','sy']/d['rho'] - data['stream','sy'][i_center]/d['rho'][i_center])*d['rho']
    d['sz'] = np.array(data['stream','sz']/d['rho'] - data['stream','sz'][i_center]/d['rho'][i_center])*d['rho']
    d['j'] = (d['x']*(d['sy'])-d['y']*(d['sx']))/d['rho']
    d['ek'] = 0.5*np.array(d['sx']**2+d['sy']**2+d['sz']**2)/d['rho'] #Energy/Volume
    d['dm'] = d['rho']*d['dV']
    d['etot'] = np.array(data['gas','etot'] + data['gas','gpot']) #total energy (including binding)
    d['egas'] = np.array(data['stream','egas'])
    d['edeg'] = A*(8*(d['rho']/B)**3*(((d['rho']/B)**2+1)**0.5-1)-\
                   ((d['rho']/B)*(2*(d['rho']/B)**2-3)*((d['rho']/B)**2+1)**0.5)+3*np.arcsinh(d['rho']/B))
    d['eint'] = np.zeros(np.shape(d['x']))
    idxs = np.where((d['egas'] - d['ek'] - d['edeg']) >= eps_1 * d['egas'])
    d['eint'][idxs] = d['egas'][idxs] - d['ek'][idxs] - d['edeg'][idxs]
    idxs = np.where(d['eint'] == 0)
    d['eint'][idxs] = d['tau'][idxs]**(5/3)
    d['primary_core'] = np.array((data['stream','rho_1'])/data['stream','density'])
    d['primary_envelope'] = np.array((data['stream','rho_2'])/data['stream','density'])
    d['secondary'] = np.array((data['stream','rho_3']+data['stream','rho_4'])/data['stream','density'])
                     


print('Spherical Averaging')

#Serialized averaging function (WAY faster)
@njit
def super_average(arr, weight_these, bins, weight_index = 1, log = True):
    rs = arr[:,0]
    weights = arr[:,weight_index]
    rr = bins
    if log:
        logrmax = np.log10(rr[-1])
        dr = np.log10(rr[1]) - np.log10(rr[0])
    else:
        logrmax = rr[-1]
        dr = rr[1] - rr[0]
    num = np.shape(arr)[1]
    
    result = np.zeros((len(bins),num))
    result[:,0] = rr
    if log:
        for i,r in enumerate(np.log10(rs+1)):
            for j in range(1,num):
                if r <= logrmax:
                    n = int(r/dr+1)
                    if j in weight_these:
                        result[n,j] += arr[i,j]*weights[i]
                    else:
                        result[n,j] += arr[i,j]
    else:
        for i,r in enumerate(rs):
            for j in range(1,num):
                if r <= logrmax:
                    n = int(r/dr+1)
                    if j in weight_these:
                        result[n,j] += arr[i,j]*weights[i]
                    else:
                        result[n,j] += arr[i,j]

    for j in weight_these:
        result[:,j] = result[:,j] / result[:,weight_index]
    
    return result

#A more user friendly function that uses the serialized version
def make_average(data, keys, weighted_keys, bin_by, resolution = 1000, ave = {}, weight = 'dV', rmax = -1, log = True):
    i_weight = keys.index(weight)
    if rmax < 0: rmax = np.max(d[bin_by]/np.sqrt(3))
    if log:
        bins = np.logspace(0,np.log10(rmax),resolution)
    else:
        bins = np.linspace(0,rmax,resolution)
    weight_these = [keys.index(this_one) for this_one in weighted_keys]
    
    arr = np.vstack([data[key] for key in keys]).T
    result = super_average(arr, weight_these, bins, weight_index = i_weight, log = log)
    result = np.delete(result,np.where(result[:,1] == 0)[0],axis=0)
    
    
    for i,key in enumerate(keys):
        ave[key] = result.T[i]

#Find max radius defined by the average radius at which material is bound
def find_rmax(data):
    temp = {}
    make_average(data, ['r','dV','dm','ek','eint'], ['ek','eint'], bin_by = 'r', ave = temp, weight = 'dV')
    
    temp['rho'] = temp['dm'] / temp['dV']
    temp['mr'] = np.array([sum(temp['dm'][:i+1]) for i in range(len(temp['dm']))])
    temp['r'] = ((3/4) * np.cumsum(temp['dV']) / np.pi)**(1/3)
    
    uint = np.array([np.trapz(temp['r'][i:]*temp['rho'][i:],temp['r'][i:]) for i in range(len(temp['r']))])
    temp['u'] = -G*temp['mr']/temp['r']-4*np.pi*G*uint
    
    temp['etot'] = temp['ek']+temp['u']*temp['rho']+temp['eint']
    
    rmax = temp['r'][np.where(np.diff(np.sign(temp['etot'])))[0][-1]]
    
    return rmax


#Initialize dictionaries to store spherically or cylindrically averaged data
ave = {}
ave_cyl = {}
rmax = find_rmax(d) 
d['j'][d['r'] > rmax] = 0 #Ensures we don't count j for cells above and below the star
make_average(d, ['r','dV','dm','tau','ek','eint','egas','edeg','j','secondary', 'primary_envelope', 'primary_core'],['tau','ek','eint','egas','edeg','j','secondary', 'primary_envelope', 'primary_core'],\
             bin_by = 'r', ave = ave, rmax = rmax)
make_average(d, ['R','dV','dm','j'], ['j'], bin_by = 'R', ave = ave_cyl, resolution = 130, rmax=rmax,\
             weight = 'dm', log = True)

#Compute other parameters using averaged parameters
ave['rho'] = ave['dm'] / ave['dV']
ave['mr'] = np.array([sum(ave['dm'][:i+1]) for i in range(len(ave['dm']))])
ave['q'] = 1-(ave['mr']/sum(ave['dm']))
ave_cyl['mr'] = np.array([sum(ave_cyl['dm'][:i+1]) for i in range(len(ave_cyl['dm']))])
ave_cyl['q'] = 1-(ave_cyl['mr']/sum(ave_cyl['dm']))

#Elements we use in BG nuclear network (mesa_28.net)
eles = ['neut', 'h1', 'h2', 'he3', 'he4', 'li7', 'be7', 'be9', 'be10', 'b8', 'c12', 'c13',\
        'n13', 'n14', 'n15', 'o14', 'o15', 'o16', 'o17', 'o18', 'f17', 'f18', 'f19', \
        'ne18', 'ne19', 'ne20', 'ne21', 'ne22']

NELE = len(eles)

#Use helmholtz EoS to compute thermodynamic variables
h = helmholtz.helmeos_DE(dens=ave['rho'], ener=ave['eint'], abar=1, zbar=1, tguess = 1e7)

ave['temp'] = h.temp
ave['entropy'] = h.stot
ave['pressure'] = h.ptot

#Initialize abundance by single star run
pabund = r.makeabund("/Users/bradmunson/rcb/3dim/bg/ms_to_wd/profile_r50.data")
p = r.profile2dict("/Users/bradmunson/rcb/3dim/bg/ms_to_wd/profile_r50.data")
p['q'] = 1 - p['q']
abund_alt = np.array([p[ele] for ele in eles]).T
abund_alt_data = np.vstack((p['q'],abund_alt.T)).T
header = str(len(p['q']))+' '+str(NELE)
np.savetxt('abund_r50.dat',abund_alt_data,header=header,comments='')

#Write AM data in MESA format
am_data = np.vstack((ave_cyl['q'],ave_cyl['j'])).T
am_data = am_data[::-1]
header = str(len(ave_cyl['q']))
np.savetxt('am_5o.dat',am_data,header=header,comments='')

#Stitch entropy profiles together (single MESA star for core + OT envelope)
p1 = r.profile2dict('ms_to_wd/profile_r50.data')
p1['q'] = 1-p1['q']
plt.semilogy(p1['q'],10**p1['logS'],label='MESA')
plt.semilogy(ave['q'],ave['entropy'],label='Octo-Tiger')

q_transition = 0.494
i_OT = np.argmin(abs(q_transition - ave['q']))
i_MESA = np.argmin(abs(q_transition - p1['q']))
new_entropy = np.concatenate((10**p1['logS'][i_MESA:][::-1],ave['entropy'][i_OT+1:]))
new_q = np.concatenate((p1['q'][i_MESA:][::-1],ave['q'][i_OT+1:]))

plt.semilogy(new_q,new_entropy,label="Frankenstein's Monster")
plt.legend()

entropy_data = np.vstack((new_q,new_entropy)).T
entropy_data = entropy_data[::-1]
header = str(len(new_q))
np.savetxt('combined_entropy.dat',entropy_data,header=header,comments='')


