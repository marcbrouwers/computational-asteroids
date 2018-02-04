from __future__ import division
from numba import cuda
import math as m
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

    
@cuda.jit
def fast(xast, xsun, xjup, vast, vsun, vjup, astforce):
    
    G = 0.000267008433813
    Msun = 1047.945205
    dt = 1./365.25
    
    row, col = cuda.grid(2)
    if row < astforce.shape[0] and col < astforce.shape[1]:
        astforce[row, col] = G*(M_sun*(xsun[0,col]-xast[row,col])/(m.sqrt((xsun[0,0]-xast[row,0])**2+(xsun[0,1]-xast[row,1])**2+(xsun[0,2]-xast[row,2])**2)**3 )+(xjup[0,col]-xast[row,col])/(m.sqrt((xjup[0,0]-xast[row,0])**2+(xjup[0,1]-xast[row,1])**2+(xjup[0,2]-xast[row,2])**2)**3 )) 
        vast[row,col] = vast[row,col] + astforce[row,col]*dt
        xast[row,col] = xast[row,col] + vast[row,col]*dt
    
    cuda.syncthreads()
    
    col2 = cuda.grid(1)
    if col2 < xsun.shape[1]:
        vsun[col2] += dt * G * (xjup[0,col2]-xsun[0,col2])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )   
        vjup[col2] -= Msun * dt * G * (xjup[0,col2]-xsun[0,col2])/(m.sqrt((xjup[0,0]-xsun[0,0])**2+(xjup[0,1]-xsun[0,1])**2+(xjup[0,2]-xsun[0,2])**2)**3 )   
        xsun[0,col2] += vsun[col2]*dt
        xjup[0,col2] += vjup[col2]*dt
        
        
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

xsun = np.array([0.,0.,0.], 'f')
vsun = np.array([0,-(V_jup*M_jup)/M_sun,0], 'f') #x,y,z,vx, vy, vz
xjup = np.array([R_jup,0,0], 'f')
vjup = np.array([0,V_jup,0], 'f')

theta = 2*m.pi*np.random.rand(1,nbodies).astype('f')[0]      #Phase of the orbits
r = np.random.uniform(R_earth, high=R_jup, size=nbodies).astype('f')
v = np.sqrt(G*M_sun/r) # this is pure keplerian, no random velocities
z = np.zeros(nbodies, 'f') # for now no z_values, otherwise make phi variable
xast = np.stack((r*np.cos(theta),r*np.sin(theta), z), axis=-1)
vast = np.stack((v*np.cos(theta+m.pi/2.),v*np.sin(theta+m.pi/2.), z), axis=-1)

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
blockspergrid_x = int(m.ceil(xast.shape[0] / threadsperblock[0]))
blockspergrid_y = int(m.ceil(np.array([xsun]).shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


start = timer()
# Start the kernel 
print "Number of intervals: ", t_intervals
time = timer()

time_int = 0

for i in range(t_intervals):

    interval = timer()
    fast[blockspergrid, threadsperblock](xast_global_mem, xsun_global_mem, xjup_global_mem, vast_global_mem, vsun_global_mem, vjup_global_mem, astforce_global_mem)
    time_int+= timer() - interval

    if (i%(t_intervals*0.01)==0) and i !=0:
        print str(round(i*100./t_intervals,2))+"%" + '  dt:  ' + str(round((timer() - time) /  i * (t_intervals - i),0))

xast = xast_global_mem.copy_to_host() 
xsun = xsun_global_mem.copy_to_host() 
xjup = xjup_global_mem.copy_to_host() 

duration = timer() - start 
print 'actual simulation time: ' + str(duration)
print 'integration', time_int, 'other', duration - time_int

plt.plot(xast[:,0], xast[:,1], 'o', c = 'g', markersize = 3)  
plt.plot(xsun[0,0], xsun[0,1], 'o', c = 'y') 
plt.plot(xjup[0,0], xjup[0,1], 'o', c = 'b')
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

plt.hist(np.linalg.norm(xast,axis=1),bins = np.linspace(0.2,1,100))
plt.show()
