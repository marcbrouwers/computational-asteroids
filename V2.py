from numba import cuda
import numpy as np
from timeit import default_timer as timer
from numba import vectorize
import numpy.random as rn
import math as m
import matplotlib.pyplot as plt
from numba import jit

# =============================================================================
# @jit(nopython = True)
# def (vec1, vec2, result):
#     """ Calculate the difference of two 3d vectors and store the result in the third parameter. """
#     result[0] = vec1[0] - vec2[0]
#     result[1] = vec1[1] - vec2[1]
#     result[2] = vec1[2] - vec2[2] 
#  
# 
# 
# 
# @jit(nopython = True)
# def div2(a, result):
#     """ Divide a 3d vector by a scalar and store the result in the second parameter."""
#     result[0] = result[0] / a
#     result[1] = result[1] / a
#     result[2] = result[2] / a
# 
# @jit(nopython=True)
# def norm(vec):
#     """ Calculate the norm of a 3d vector. """
#     return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])
# 
# 
# @vectorize(['float64(float64, float64)'], target='cuda')
# def fuerza(a, b):
#     return a * b / (a+b)
# =============================================================================


def main(): 
    def init():
        xsun = np.array([0,0,0])
        vsun = np.array([0,-(V_jup*M_jup)/M_sun,0]) #x,y,z,vx, vy, vz
        xjup = np.array([R_jup,0,0])
        vjup = np.array([0,V_jup,0])
        
        theta = 2*m.pi*np.random.rand(1,nbodies)[0]      #Phase of the orbits
        r = np.random.uniform(R_earth, high=R_jup, size=nbodies)
        v = np.sqrt(G*M_sun/r) # this is pure keplerian, no random velocities
        z = np.zeros(nbodies) # for now no z_values, otherwise make phi variable
        xast = np.stack((r*np.cos(theta),r*np.sin(theta), z), axis=-1)
        vast = np.stack((v*np.cos(theta+m.pi/2),v*np.sin(theta+m.pi/2), z), axis=-1)
        return xast, vast, xsun, vsun, xjup, vjup

    
    def EulerCromer_gravitational(xobj,vobj,xsun, xjup, dt):
        vobj = vobj + acceleration(xobj,xsun, xjup) * dt
        xobj = xobj + vobj * dt
        return xobj, vobj
    
    def acceleration(xobj,xsun, xjup):
        if np.array_equal(xobj,xsun):
            return G * M_jup * (xjup - xobj)/(np.linalg.norm(xjup - xobj)**3)
        elif np.array_equal(xobj,xjup):
            return G * M_sun * (xsun - xobj)/(np.linalg.norm(xsun - xobj)**3)
        else:
            return G *(M_sun* (xsun - xobj)/((np.linalg.norm(xsun - xobj,axis=1)**3)[:,None]) + M_jup* (xjup - xobj)/((np.linalg.norm(xjup - xobj,axis=1)**3)[:,None]))
    

    def integrator(xast, vast, xsun, vsun, xjup, vjup):
        time = timer()
        for i in range(t_intervals):
            xast, vast = EulerCromer_gravitational(xast,vast,xsun, xjup, dt)
            vast = vast[np.amax(np.absolute(xast[:]),axis=1)<10*R_jup]
            xast = xast[np.amax(np.absolute(xast[:]),axis=1)<10*R_jup]
            xsun, vsun = EulerCromer_gravitational(xsun,vsun,xsun, xjup, dt)
            xjup, vjup = EulerCromer_gravitational(xjup,vjup,xsun, xjup, dt)
            if (i%(t_intervals/100)==0) and i !=0:
                print str(i*100./t_intervals)+"%" + '  dt:  ' + str((timer() - time) /  i * (t_intervals - i)), xast.shape

    M_jup_scaling = 1.898e+27 # normalization factor for mass
    R_jup_scaling = 7.785e+11 # normalization factor for length
    year_scaling = 365 * 24 * 3600 # normalization factor for time
    G = 6.67408e-11 * R_jup_scaling**(-3) * M_jup_scaling * year_scaling**2
    M_sun = 1.989e+30 / M_jup_scaling
    R_jup = 1.
    M_jup = 0.
    V_jup = G*M_sun/ R_jup
    R_earth = 1.496e+11 / R_jup_scaling
    
    nbodies = 2500
    dt = 1./365 # in years
    t_end = 100 # in years
    t_intervals = int(t_end / dt)

    
    
    start = timer()
    xast, vast, xsun, vsun, xjup, vjup = init()
    integrator(xast, vast, xsun, vsun, xjup, vjup)       
    duration = timer() - start   
    print 'actual simulation time: ' + str(duration)
    
    print 'plotting...'
    plt.figure(figsize=(8,6))
    plt.plot(xsun[0], xsun[1], 'o', c = 'y') 
    plt.plot(xjup[0], xjup[1], 'o', c = 'b') 
    plt.plot(xast[:,0], xast[:,1], 'o', c = 'g', markersize = 1)  
    plt.show()


if __name__ == '__main__':
    main()