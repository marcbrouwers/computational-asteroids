# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 23:01:42 2018

@author: Javier
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:54:35 2018

@author: Javier
"""
from numba import cuda
import numpy as np
from timeit import default_timer as timer
from numba import vectorize
import numpy.random as rn
import math as m
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython = True)
def sub3(vec1, vec2, result):
    """ Calculate the difference of two 3d vectors and store the result in the third parameter. """
    result[0] = vec1[0] - vec2[0]
    result[1] = vec1[1] - vec2[1]
    result[2] = vec1[2] - vec2[2] 
 



@jit(nopython = True)
def div2(a, result):
    """ Divide a 3d vector by a scalar and store the result in the second parameter."""
    result[0] = result[0] / a
    result[1] = result[1] / a
    result[2] = result[2] / a

@jit(nopython=True)
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2])


@vectorize(['float64(float64, float64)'], target='cuda')
def fuerza(a, b):
    return a * b / (a+b)


def main():    
      
    def EulerCromer_gravitational(xast,vast,xsun, xjup, dt):
        vast = vast + force(xast,xsun, xjup) * dt
        xast = xast + vast * dt
        return xast, vast
    
#    def f12(Body1,Body2,position_1):
#        if Body1 == Body2:
#            return np.array([0,0,0])
#        pos = position_1-positions[Body2][0:3]
#        dist = np.linalg.norm(pos)
#        return -G* positions[Body2][-1]*pos/(dist)**3

    def force(xast,xsun, xjup):
        if np.array_equal(xast,xsun):
            return G * M_jup * (xjup - xast)/(np.linalg.norm(xjup - xast)**3)
        elif np.array_equal(xast,xjup):
            return G * M_sun * (xsun - xast)/(np.linalg.norm(xsun - xast)**3)
#        F = 0
#        Fsun = -G *M_sun* (xsun - xast)/((np.linalg.norm(xsun - xast,axis=1)**3)[:,None])
#        Fjup = -G *M_jup* (xjup - xast)/((np.linalg.norm(xjup - xast,axis=1)**3)[:,None])
#        F = Fsun + Fjup
        else:
            return G *(M_sun* (xsun - xast)/((np.linalg.norm(xsun - xast,axis=1)**3)[:,None]) + M_jup* (xjup - xast)/((np.linalg.norm(xjup - xast,axis=1)**3)[:,None]))
    

    def integrator(xast, vast, xsun, vsun, xjup, vjup):
#        plt.figure(figsize=(8,6))
        for i in range(t_intervals):
            xast, vast = EulerCromer_gravitational(xast,vast,xsun, xjup, dt)
            xast = xast[np.amax(np.absolute(xast[:]),axis=1)<10*R_jup]
            vast = vast[np.amax(np.absolute(xast[:]),axis=1)<10*R_jup]
            xsun, vsun = EulerCromer_gravitational(xsun,vsun,xsun, xjup, dt)
            xjup, vjup = EulerCromer_gravitational(xjup,vjup,xsun, xjup, dt)
            if (i%(t_intervals/1000)==0): print str(i*100./t_intervals)+"%"
                #Condition to get rid of asteroids flying away
#                if positions[Body][0] > 4*R_jup or positions[Body][1] > 4* R_jup or positions[Body][0] < -4*R_jup or positions[Body][1] < -4* R_jup:  #Eliminating from the list asteroids scaping
#                    bodieslist.remove(Body)
#            if (i % (100)) == 0:
#                plt.plot(xsun[0],xsun[1], 'o', c = 'y') 
#                plt.plot(xjup[0], xjup[1], 'o', c = 'b') 
#                if(nbodies!=0): 
#                    for k in range(nbodies):
#                        plt.plot(xast[k,0], xast[k,1], 'o', c = 'g')  
#
#        plt.show()
        return xast, vast, xsun, vsun, xjup, vjup
#            if (i%(t_intervals/1000)==0): print str(i*100/t_intervals)+"%"





    G = 6.67e-11
    R_jup = 7.785e+11
    M_jup = 1.898e+27
    V_jup = 1.3058e+4
    M_sun = 1.989e+30
    R_earth = 1.496e+11 #Smallest R for an asteroid
    

    
    t_intervals = int(1e3)
    dt = 1e5

    xsun = np.array([0,0,0])
    vsun = np.array([0,-(V_jup*M_jup)/M_sun,0]) #x,y,z,vx, vy, vz
    xjup = np.array([R_jup,0,0])
    vjup = np.array([0,V_jup,0])

    nbodies = 1000
    xast = np.empty((nbodies,3))
    vast = np.empty((nbodies,3))
    for i in range(nbodies):
        theta = 2*m.pi*rn.random()        #Phase of the orbit
        r = rn.uniform(R_earth,R_jup)    #Radius of the asteroids between earth and jupiter orbit
        v = m.sqrt(G*M_sun/r) * rn.uniform(1, 1.01)
        xast[i,:]=[r*m.cos(theta),r*m.sin(theta),0]
        vast[i,:]=[v*m.cos(theta+m.pi/2),v*m.sin(theta+m.pi/2),0]
    
    start = timer()     
    xast, vast, xsun, vsun, xjup, vjup = integrator(xast, vast, xsun, vsun, xjup, vjup)       
    duration = timer() - start   
    print(duration)
    
    plt.figure(figsize=(8,6))
    plt.plot(xsun[0], xsun[1], 'o', c = 'y') 
    plt.plot(xjup[0], xjup[1], 'o', c = 'b') 
    for k in range(len(xast[:,0])):
        plt.plot(xast[k,0], xast[k,1], 'o', c = 'g', markersize = 1)  
    plt.show()

#    start = timer()  
#    xast = np.ones((nbodies,3))
#    xsun = np.array([0,0,1])
#    print  (xsun - xast)
#    print  np.linalg.norm(xsun - xast,axis=1)
#    print( np.linalg.norm(xsun - xast,axis=1)**3)
#    print (xsun - xast)/((np.linalg.norm(xsun - xast,axis=1)**3)[:,None])
#    c = xsun - xast
#    duration = timer() - start   
#    print(c)

if __name__ == '__main__':
    main()