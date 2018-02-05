# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 15:13:31 2018

@author: Javier
"""

from __future__ import division
from numba import cuda
import math
import math as m
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle

#------------------------------  Integration functions and saving/loading  ------------------------------
@cuda.jit
def fast(xast, xsun, xjup, vast, vsun, vjup, astforce):
    """Perform matrix multiplication of C = A * B
    """

    dt = 0.5/365.25
    row, col = cuda.grid(2)
    if row < astforce.shape[0] and col < astforce.shape[1]:
        astforce[row, col] = 0.0376291660908*(1047.945205*(xsun[0,col]-xast[row,col])/(m.sqrt((xsun[0,0]-xast[row,0])**2+(xsun[0,1]-xast[row,1])**2+(xsun[0,2]-xast[row,2])**2)**3 )+(xjup[0,col]-xast[row,col])/(m.sqrt((xjup[0,0]-xast[row,0])**2+(xjup[0,1]-xast[row,1])**2+(xjup[0,2]-xast[row,2])**2)**3 )) 
        vast[row,col] = vast[row,col] + astforce[row,col]*dt
        xast[row,col] = xast[row,col] + vast[row,col]*dt
    cuda.syncthreads()


@cuda.jit
def fstel(xsun, vsun, xjup, vjup):
 
    dt = 0.5/365.25 
    cuda.syncthreads()    
    col = cuda.grid(1)
    if col < xsun.shape[1]:
        vsun[col] += dt * 0.0376291660908 * (xjup[0,col]-xsun[0,col])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )   
        vjup[col] -= 1047.945205   * dt * 0.0376291660908 * (xjup[0,col]-xsun[0,col])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )    
        cuda.syncthreads()
        xsun[0,col] += vsun[col]*dt
        xjup[0,col] += vjup[col]*dt


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'wb') as f:
        pickle.load(f)


#------------------------------  Constants/variables  ------------------------------
        
M_jup_scaling = 1.898e+27 # normalization factor for mass
AU_scaling = 1.495978707e+11 # normalization factor for length
year_scaling = 365 * 24 * 3600 # normalization factor for time
G = 6.67408e-11 * AU_scaling**(-3) * M_jup_scaling * year_scaling**2
M_sun = 1.989e+30 / M_jup_scaling
R_jup = 5.4588
M_jup = 0.
e = 0.04839266
V_jup = np.sqrt((G*M_sun/ R_jup)*((1+e)/(1-e)))

nbodies = 1000                                                                 
dt = 0.5/365.25 # in years
t_end = 100 # in years
t_intervals = int(t_end / dt)


#------------------------------  Initial conditions  ------------------------------

xsun = np.array([0.,0.,0.], 'f')
vsun = np.array([0,-(V_jup*M_jup)/M_sun,0], 'f') #x,y,z,vx, vy, vz
xjup = np.array([R_jup,0,0], 'f')
vjup = np.array([0,V_jup,0], 'f')

theta = np.random.uniform(0, high=2*np.pi, size=nbodies).astype('f')     #Phase of the orbits
phi = np.full_like(theta, m.pi/2.) + np.random.uniform(0, high=m.pi/100., size=nbodies).astype('f')
phi = np.full_like(theta, m.pi/2.)
r = np.linspace(2, 3.5, num=nbodies).astype('f')
v = np.sqrt(G*M_sun/r) * np.random.uniform(0.95, high=1.05, size=nbodies).astype('f')
xast = np.stack((r*np.cos(theta)*np.sin(phi),r*np.sin(theta)*np.sin(phi), r*np.cos(phi)), axis=-1)
vast = np.stack((v*np.cos(theta+m.pi/2.)*np.sin(phi+m.pi/2.),v*np.sin(theta+m.pi/2.)*np.sin(phi+m.pi/2.),v*np.cos(phi+m.pi/2.)), axis=-1)


# Copy the arrays to the device
xast_global_mem = cuda.to_device(xast)
xsun_global_mem = cuda.to_device(np.array([xsun]))
xjup_global_mem = cuda.to_device(np.array([xjup]))
vast_global_mem = cuda.to_device(vast)
vsun_global_mem = cuda.to_device(vsun)
vjup_global_mem = cuda.to_device(vjup)
# Allocate memory on the device for the result
astforce_global_mem = cuda.device_array((nbodies, 3))
sunforce_global_mem = cuda.device_array((1, 3))

v_ast = np.linalg.norm(vast,axis=1)
r_ast = np.linalg.norm(xast,axis=1)
semi_ast = G*M_sun*r_ast/(2.*G*M_sun - v_ast**2 * r_ast)

#------------------------------  Integration  ------------------------------
# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(xast.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(np.array([xsun]).shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

blockspergrid_x1 = int(math.ceil(xast.shape[0] / threadsperblock[0]))
blockspergrid_y1 = int(math.ceil(np.array([xsun]).shape[1] / threadsperblock[1]))
blockspergrid1 = (blockspergrid_x1, blockspergrid_y1)


start = timer()

#
# Start the kernel 
print "Number of intervals: ", t_intervals
time = timer()

time_1 = 0
time_2 = 0
for i in range(t_intervals):
    interval = timer()
    fast[blockspergrid, threadsperblock](xast_global_mem, xsun_global_mem, xjup_global_mem, vast_global_mem, vsun_global_mem, vjup_global_mem, astforce_global_mem)
    time_1+= timer() - interval
    
    interval = timer()
    fstel[blockspergrid, threadsperblock](xsun_global_mem, vsun_global_mem, xjup_global_mem, vjup_global_mem)
    time_2 += timer() - interval
    
    if (i%(t_intervals/100)==0) and i !=0:
                print str(round(i*100./t_intervals,2))+"%" + '  dt:  ' + str(round((timer() - time) /  i * (t_intervals - i),0)), xast.shape, vast.shape
                xast = xast_global_mem.copy_to_host()           
                vast = vast_global_mem.copy_to_host() 
                
                xast_global_mem = cuda.to_device(xast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup])
                vast_global_mem = cuda.to_device(vast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup])
                
                astforce_global_mem = cuda.device_array((len(xast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup]), 3))

duration = timer() - start 
print 'actual simulation time: ' + str(duration)
print 'asteroids', time_1, 'planets', time_2, 'other', duration - time_1 - time_2


#------------------------------  Saving/loading  ------------------------------

xast = xast_global_mem.copy_to_host()
vast = vast_global_mem.copy_to_host()
xsun = xsun_global_mem.copy_to_host() 
xjup = xjup_global_mem.copy_to_host()

r_ast = np.linalg.norm(xast,axis=1)
v_ast = np.linalg.norm(vast,axis=1)
semi_ast = G*M_sun*r_ast/(2.*G*M_sun - v_ast**2 * r_ast)

# =============================================================================
# save_obj(xsun, "xsun")
# save_obj(xjup, "xjup")
# save_obj(xast, "xast")
# save_obj(vsun, "vsun")
# save_obj(vjup, "vjup")
# save_obj(vast, "vast")
# save_obj(semi_ast, "semi_ast")
# 
# xast_result = load_obj('xast')
# xsun_result = load_obj('xjup')
# xjup_result = load_obj('xsun')
# vast_result = load_obj('vast')
# vsun_result = load_obj('vjup')
# vjup_result = load_obj('vsun')
# =============================================================================

#------------------------------  Plotting  ------------------------------

plt.figure(figsize=(8,6))

plt.plot(xast[:,0], xast[:,1], 'o', c = 'g', markersize = 3)  
plt.plot(xsun[0,0], xsun[0,1], 'o', c = 'y') 
plt.plot(xjup[0,0], xjup[0,1], 'o', c = 'b')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()

plt.hist(semi_ast,bins = np.linspace(2,3.5,100))
plt.show()
plt.hist(r_ast,bins = np.linspace(2,3.5,100))
plt.show()