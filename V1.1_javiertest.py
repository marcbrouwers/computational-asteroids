from __future__ import division
import numpy as np
import numpy.random as rn
import math as m
import matplotlib.pyplot as plt
<<<<<<< HEAD
from timeit import default_timer as timer
from numba import vectorize

=======
import time
import sys
from timeit import default_timer as timer
>>>>>>> 1241f6b32fae1c1b8f1a766ee0746f219d0f1a1d

def f12(Body1,Body2,position_1):
    if Body1 == Body2:
        return np.array([0,0,0])
    pos = position_1-positions[Body2][0:3]
    dist = np.linalg.norm(pos)
    return -G* positions[Body2][-1]*pos/(dist)**3

def force(Body1, position_1):
    F = 0
    for k in ["Jupiter", "Sun"]:
        F = F + f12(Body1, k, position_1)
    return F


def asteroid_init():
    
    for i in range(nbodies):
        theta = 2*m.pi*rn.random()        #Phase of the orbit
        r = rn.uniform(R_earth,R_jup)    #Radius of the asteroids between earth and jupiter orbit
        v = m.sqrt(G*M_sun/r) * rn.uniform(1, 1.01)
        bodieslist.append("Body "+str(i+1))
        positions[bodieslist[i+2]] = np.array([r*m.cos(theta),r*m.sin(theta),0,v*m.cos(theta+m.pi/2),v*m.sin(theta+m.pi/2),0, 0])
        
       
def RK4_gravitational(Body, dt):
    
    k1 = np.concatenate((positions[Body][3:6], force(Body, positions[Body][:3]))) * dt
    k2 = np.concatenate((positions[Body][3:6] + k1[3:] * 0.5, force(Body, positions[Body][:3]  + k1[:3] * 0.5))) * dt
    k3 = np.concatenate((positions[Body][3:6] + k2[3:] * 0.5, force(Body, positions[Body][:3]  + k2[:3] * 0.5))) * dt
    k4 = np.concatenate((positions[Body][3:6] + k3[3:], force(Body, positions[Body][:3]  + k3[:3]))) * dt
    positions[Body][:6] = positions[Body][:6] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
    
    return positions[Body]


def integrator():
    t_end = dt * t_intervals
    plt.figure(figsize=(8,6))
    for i in range(t_intervals):
        temporalbodielist = bodieslist
        for j, Body in enumerate(temporalbodielist):
            positions[Body] = EulerCromer_gravitational(Body, dt)
            #Condition to get rid of asteroids flying away
            if positions[Body][0] > 4*R_jup or positions[Body][1] > 4* R_jup or positions[Body][0] < -4*R_jup or positions[Body][1] < -4* R_jup:  #Eliminating from the list asteroids scaping
                bodieslist.remove(Body)
#        if (i % (100)) == 0:
#            plt.plot(positions["Sun"][0], positions["Sun"][1], 'o', c = 'y') 
        #    plt.plot(positions["Jupiter"][0], positions["Jupiter"][1], 'o', c = 'b') 
        #    if(nbodies!=0): 
        #        for k in range(nbodies):
        #            plt.plot(positions["Body "+str(k+1)][0], positions["Body "+str(k+1)][1], 'o', c = 'g')  
#    plt.show()
        if (i%(t_intervals/1000)==0): print str(i*100/t_intervals)+"%"
    return positions


def EulerCromer_gravitational(Body, dt):
    
    v_np1 = positions[Body][3:6] + force(Body, positions[Body][:3]) * dt
    x_np1 = positions[Body][:3] + v_np1 * dt
    positions[Body][:6] = np.concatenate((x_np1, v_np1))
    
    return positions[Body]

    
G = 6.67e-11
R_jup = 7.785e+11
M_jup = 1.898e+27
V_jup = 1.3058e+4
M_sun = 1.989e+30
R_earth = 1.496e+11 #Smallest R for an asteroid

<<<<<<< HEAD
nbodies = 2
=======
nbodies = 1000
>>>>>>> 1241f6b32fae1c1b8f1a766ee0746f219d0f1a1d
print "Initial number of asteroids = ",nbodies
t_intervals = int(1e3)
dt = 1e5
bodieslist = ["Sun", "Jupiter"]
initcond = np.zeros((nbodies,7))
positions = {}
positions["Sun"] = np.array([0,0,0,0,-(V_jup*M_jup)/M_sun,0,M_sun]) #x,y,z,vx, vy, vz
positions["Jupiter"] = np.array([R_jup,0,0,0,V_jup,0,M_jup])
asteroid_init()
start = timer()   
integrator()
duration = timer() - start   
print(duration)
print "Final number of asteroids = ", len(bodieslist)-2

plt.figure(figsize=(8,6))
plt.plot(positions["Sun"][0], positions["Sun"][1], 'o', c = 'y') 
plt.plot(positions["Jupiter"][0], positions["Jupiter"][1], 'o', c = 'b') 
bodieslist.remove("Sun")
bodieslist.remove("Jupiter")
for k in bodieslist:
    plt.plot(positions[k][0], positions[k][1], 'o', c = 'g', markersize = 1)  
plt.show()