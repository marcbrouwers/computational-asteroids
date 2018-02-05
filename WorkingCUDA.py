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

# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row,col] = A[row,col]+ B[row,col]
    
@cuda.jit
def fast(xast, xsun, xjup, vast, vsun, vjup, astforce):
    """Perform matrix multiplication of C = A * B
    """
    #G and Msun values used explicitly in order to not copy its value back and forth to the GPU every time they are called     

#    G = 0.000267008433813
#    Msun = 1047.945205
    dt = 0.5/365.25
    row, col = cuda.grid(2)
    if row < astforce.shape[0] and col < astforce.shape[1]:
        astforce[row, col] = 0.000267008433813*(1047.945205*(xsun[0,col]-xast[row,col])/(m.sqrt((xsun[0,0]-xast[row,0])**2+(xsun[0,1]-xast[row,1])**2+(xsun[0,2]-xast[row,2])**2)**3 )+(xjup[0,col]-xast[row,col])/(m.sqrt((xjup[0,0]-xast[row,0])**2+(xjup[0,1]-xast[row,1])**2+(xjup[0,2]-xast[row,2])**2)**3 )) 
        vast[row,col] = vast[row,col] + astforce[row,col]*dt
        xast[row,col] = xast[row,col] + vast[row,col]*dt
#    if col < sunforce.shape[1]:
#        sunforce[0,col] = G * (xjup[0,col]-xsun[0,col])/(((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**(3./2.) )   
@cuda.jit
def fstel(xsun, vsun, xjup, vjup):
    #G and Msun values used explicitly in order to not copy its value back and forth to the GPU every time they are called     
    dt = 0.5/365.25      
    col = cuda.grid(1)
    if col < xsun.shape[1]:
        vsun[col] += dt * 0.000267008433813 * (xjup[0,col]-xsun[0,col])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )   
        vjup[col] -= 1047.945205   * dt * 0.000267008433813 * (xjup[0,col]-xsun[0,col])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )    
        xsun[0,col] += vsun[col]*dt
        xjup[0,col] += vjup[col]*dt


def EulerCromer_sun(xsun,vsun, xjup, dt):
        vsun = vsun + M_jup *acceleration(xsun, xjup) * dt
        xsun = xsun + vsun * dt
        return xsun, vsun    
def EulerCromer_jup(xjup,vjup, xsun, dt):
        vjup = vjup - M_sun* acceleration(xsun, xjup) * dt
        xjup = xjup + vjup * dt
        return xjup, vjup    
def acceleration(xsun, xjup):
    return G *  (xjup - xsun)/(np.linalg.norm(xjup - xsun)**3)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
M_jup_scaling = 1.898e+27 # normalization factor for mass
R_jup_scaling = 7.785e+11 # normalization factor for length
year_scaling = 365 * 24 * 3600 # normalization factor for time
G = 6.67408e-11 * R_jup_scaling**(-3) * M_jup_scaling * year_scaling**2

M_sun = 1.989e+30 / M_jup_scaling
R_jup = 1.
M_jup = 1.
e = 0.04839266
V_jup = np.sqrt((G*M_sun/ R_jup)*((1+e)/(1-e)))
R_earth = 1.496e+11 / R_jup_scaling

nbodies = 1000                                                                    
dt = 0.5/365.25 # in years
t_end = 5 # in years
t_intervals = int(t_end / dt)
# Host code



xsun = np.array([0.,0.,0.], 'f')
vsun = np.array([0,-(V_jup*M_jup)/M_sun,0], 'f') #x,y,z,vx, vy, vz
xjup = np.array([R_jup,0,0], 'f')
vjup = np.array([0,V_jup,0], 'f')

theta = 2*m.pi*np.random.rand(1,nbodies).astype('f')[0]      #Phase of the orbits
r = np.random.uniform(R_earth, high=R_jup, size=nbodies).astype('f')
v = np.sqrt(G*M_sun/r) # this is pure keplerian, no random velocities
z = np.zeros(nbodies, 'f') # for now no z_values, otherwise make phi variable
xast = np.stack((r*np.cos(theta),r*np.sin(theta), z), axis=-1)
vast = np.stack((v*np.cos(theta+m.pi/2),v*np.sin(theta+m.pi/2), z), axis=-1)


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



# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(xast.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(np.array([xsun]).shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

blockspergrid_x1 = int(math.ceil(xast.shape[0] / threadsperblock[0]))
blockspergrid_y1 = int(math.ceil(np.array([xsun]).shape[1] / threadsperblock[1]))
blockspergrid1 = (blockspergrid_x1, blockspergrid_y1)

#plt.figure(figsize=(8,6))
#plt.plot(B[0,0], B[0,1], 'o', c = 'y') 
#plt.plot(C[0,0], C[0,1], 'o', c = 'b') 
#plt.plot(xast[:,0], xast[:,1], 'o', c = 'k', markersize = 1)  

start = timer()
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
                print str(round(i*100./t_intervals,2))+"%" + '  dt:  ' + str(round((timer() - time) /  i * (t_intervals - i),0)), xast.shape
                xast = xast_global_mem.copy_to_host()           
                vast = vast_global_mem.copy_to_host() 
                
                xast_global_mem = cuda.to_device(xast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup])
                vast_global_mem = cuda.to_device(vast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup])
                
                astforce_global_mem = cuda.device_array((len(xast[np.amax(np.absolute(xast[:]),axis=1)<2*R_jup]), 3))

#plt.show()
    
    
#    fstel[blockspergrid, threadsperblock](xsun_global_mem, vsun_global_mem, xjup_global_mem, vjup_global_mem)

# Copy the result back to the host

#astforce = astforce_global_mem.copy_to_host()
#vast = vast_global_mem.copy_to_host() 
xast = xast_global_mem.copy_to_host() 
xsun = xsun_global_mem.copy_to_host() 
xjup = xjup_global_mem.copy_to_host() 

duration = timer() - start 
print 'actual simulation time: ' + str(duration)
print 'asteroids', time_1, 'planets', time_2, 'other', duration - time_1 - time_2

save_obj(xsun, "xsun")
save_obj(xjup, "xjup")
save_obj(xast, "xast")

plt.figure(figsize=(8,6))

plt.plot(xast[:,0], xast[:,1], 'o', c = 'g', markersize = 3)  
plt.plot(xsun[0,0], xsun[0,1], 'o', c = 'y') 
plt.plot(xjup[0,0], xjup[0,1], 'o', c = 'b')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.show()
#
#plt.hist(np.linalg.norm(xast,axis=1),bins = np.linspace(0.2,1,1000))
#plt.show()
print xjup
