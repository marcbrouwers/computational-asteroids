# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 15:13:31 2018

@author: Javier
"""

from __future__ import division
from numba import cuda
import numba
import math
import math as m
import numpy as np
from numba import jit
from timeit import default_timer as timer
import matplotlib.pyplot as plt

@jit(nopython=True)
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row,col] = A[row,col]+ B[row,col]
    
@cuda.jit
def fast(xast, xsun, xjup, vast, vsun, vjup, astforce, sunforce):
    """Perform matrix multiplication of C = A * B
    """
    G = 0.000267008433813
    Msun = 1047.945205
    dt = 1/365.25
    row, col = cuda.grid(2)
    if row < astforce.shape[0] and col < astforce.shape[1]:
        astforce[row, col] = G*(Msun*(xsun[0,col]-xast[row,col])/(((xsun[0,0]-xast[row,0])**2+(xsun[0,1]-xast[row,1])**2+(xsun[0,2]-xast[row,2])**2)**(3./2.) )+(xjup[0,col]-xast[row,col])/(((xjup[0,0]-xast[row,0])**2+(xjup[0,1]-xast[row,1])**2+(xjup[0,2]-xast[row,2])**2)**(3./2.) )) 
        vast[row,col] = vast[row,col] + astforce[row,col]*dt
        xast[row,col] = xast[row,col] + vast[row,col]*dt
#    if col < sunforce.shape[1]:
#        sunforce[0,col] = G * (xjup[0,col]-xsun[0,col])/(((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**(3./2.) )   
@cuda.jit
def fstel(xsun, vsun, xjup, vjup):  
    Msun = 1047.945205      
    dt = 1/365.25      
    
    sunforcex = G * (xjup[0,0]-xsun[0,0])/(((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**(3./2.) )   
    sunforcey = G * (xjup[0,1]-xsun[0,1])/(((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**(3./2.) )   
    sunforcez = G * (xjup[0,2]-xsun[0,2])/(((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**(3./2.) )   
    vsun[0] += sunforcex*dt
    vjup[0] -= Msun * sunforcex*dt
    vsun[1] +=  sunforcey*dt
    vjup[1] -=  Msun * sunforcey*dt 
    vsun[2] +=  sunforcez*dt
    vjup[2] -=  Msun * sunforcez*dt
    xsun[0,0] = xsun[0,0] + vsun[0]*dt
    xjup[0,0] = xjup[0,0] + vjup[0]*dt
    xsun[0,1] = xsun[0,1] + vsun[1]*dt
    xjup[0,1] = xjup[0,1] + vjup[1]*dt 
    xsun[0,2] = xsun[0,2] + vsun[2]*dt
    xjup[0,2] = xjup[0,2] + vjup[2]*dt

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

        
M_jup_scaling = 1.898e+27 # normalization factor for mass
R_jup_scaling = 7.785e+11 # normalization factor for length
year_scaling = 365 * 24 * 3600 # normalization factor for time
G = 6.67408e-11 * R_jup_scaling**(-3) * M_jup_scaling * year_scaling**2

M_sun = 1.989e+30 / M_jup_scaling
R_jup = 1.
M_jup = 1.
V_jup = np.sqrt(G*M_sun/ R_jup)
R_earth = 1.496e+11 / R_jup_scaling

nbodies = 10000
dt = 1./365.25 # in years
t_end = 100 # in years
t_intervals = int(t_end / dt)
# Host code



xsun = np.array([0.,0.,0.])
vsun = np.array([0,-(V_jup*M_jup)/M_sun,0]) #x,y,z,vx, vy, vz
xjup = np.array([R_jup,0,0])
vjup = np.array([0,V_jup,0])

theta = 2*m.pi*np.random.rand(1,nbodies)[0]      #Phase of the orbits
r = np.random.uniform(R_earth, high=R_jup, size=nbodies)
v = np.sqrt(G*M_sun/r) # this is pure keplerian, no random velocities
z = np.zeros(nbodies) # for now no z_values, otherwise make phi variable
xast = np.stack((r*np.cos(theta),r*np.sin(theta), z), axis=-1)
vast = np.stack((v*np.cos(theta+m.pi/2),v*np.sin(theta+m.pi/2), z), axis=-1)
# Initialize the data arrays
C = np.array([xjup])#numpy.full((24, 12), 3, numpy.float) # matrix containing all 3's
A = xast#numpy.array(([2.,1.,3.],[1.,3.,5.])) # matrix containing all 4's
B = np.array([xsun])

# Copy the arrays to the device
xast_global_mem = cuda.to_device(A)
xsun_global_mem = cuda.to_device(B)
xjup_global_mem = cuda.to_device(C)
vast_global_mem = cuda.to_device(vast)
vsun_global_mem = cuda.to_device(vsun)
vjup_global_mem = cuda.to_device(vjup)
# Allocate memory on the device for the result
astforce_global_mem = cuda.device_array((nbodies, 3))
sunforce_global_mem = cuda.device_array((1, 3))

# =============================================================================
# # Set the number of threads in a block
# threadsperblock = 32 
# 
# # Calculate the number of thread blocks in the grid
# blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
# print 
# =============================================================================

# Configure the blocks
threadsperblock = (16, 16)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
#blockspergrid = (10,10)
#print blockspergrid_x, blockspergrid_y


#plt.figure(figsize=(8,6))
#plt.plot(B[0,0], B[0,1], 'o', c = 'y') 
#plt.plot(C[0,0], C[0,1], 'o', c = 'b') 
#plt.plot(xast[:,0], xast[:,1], 'o', c = 'k', markersize = 1)  

start = timer()
# Start the kernel 
print "Number of intervals: ", t_intervals
time = timer()
time_gpu = 0
time_comp = 0
time_send = 0
time_receive = 0
for i in range(t_intervals):

    time_interval = timer()
    fast[blockspergrid, threadsperblock](xast_global_mem, xsun_global_mem, xjup_global_mem, vast_global_mem, vsun_global_mem, vjup_global_mem, astforce_global_mem, sunforce_global_mem)
    time_gpu += timer() - time_interval

    
    time_interval = timer()
    xsun = xsun_global_mem.copy_to_host() 
    xjup = xjup_global_mem.copy_to_host()
    vsun = vsun_global_mem.copy_to_host()
    vjup = vjup_global_mem.copy_to_host()
    time_send += timer() - time_interval
    
    time_interval = timer()
    xsun, vsun = EulerCromer_sun(xsun,vsun, xjup, dt)
    xjup, vjup = EulerCromer_jup(xjup,vjup, xsun, dt)
    time_comp += timer() - time_interval

    
    time_interval = timer()
    xsun_global_mem = cuda.to_device(xsun)
    vsun_global_mem = cuda.to_device(vsun)
    xjup_global_mem = cuda.to_device(xjup)
    vjup_global_mem = cuda.to_device(vjup)
    time_receive += timer() - time_interval
    
    if (i%(t_intervals/100)==0) and i !=0:
                print str(round(i*100./t_intervals,2))+"%" + '  dt:  ' + str(round((timer() - time) /  i * (t_intervals - i),0)), xast.shape

print 'gpu', time_gpu, 'comp', time_comp, 'receive', time_receive, 'send', time_send
#plt.show()
    
    
#    fstel[blockspergrid, threadsperblock](xsun_global_mem, vsun_global_mem, xjup_global_mem, vjup_global_mem)

# Copy the result back to the host

#astforce = astforce_global_mem.copy_to_host()
#vast = vast_global_mem.copy_to_host() 
xast = xast_global_mem.copy_to_host() 

#xsun = xsun_global_mem.copy_to_host() 
#xjup = xjup_global_mem.copy_to_host() 

duration = timer() - start 
print 'actual simulation time: ' + str(duration)


#plt.figure(figsize=(8,6))

plt.plot(xast[:,0], xast[:,1], 'o', c = 'g', markersize = 3)  
plt.plot(xsun[0,0], xsun[0,1], 'o', c = 'y') 
plt.plot(xjup[0,0], xjup[0,1], 'o', c = 'b')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.show()

plt.hist(np.linalg.norm(xast,axis=1),bins = np.linspace(0.2,1,100))
plt.show()

