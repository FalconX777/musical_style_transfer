import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.optimize
from scipy import signal
import datetime, os
from os import walk

import MIDI_handling, sequence_alignment

def d_freq(k,l):
    return abs(k-l)

def d_time(i,j,lambd=1):
    return lambd*abs(i-j)

def d_tot(a,b):
    return m.sqrt(d_time(a[0],b[0])**2 + d_freq(a[1],b[1])**2)

def entropy(mu,a):
    h = -a*mu*np.log(mu)
    return np.nansum(h)

def entropic_sharpening(mu,H_0,a):
    if entropy(mu,a) + np.sum(a*mu) > H_0 + 1:
        try:
            beta = abs(sp.optimize.fsolve(lambda b: np.linalg.norm(entropy(mu**abs(b),a) + np.sum(a*mu**abs(b)) - (H_0 + 1)),1)[0])
        except:
            beta = 1
    else :
        beta = 1
    return mu**beta

# Exact Kernel Barycenter (see Cuturi)

def ker_bar_exa(s,alpha,gamma=1,lambd=1,epoch_max=10,sharpening=True): # shp for shape of the considered space, each s[i] has to have this shape (complete with 0 if necessary)
    shp = s[0].shape
    k = len(s)
    if k == 1:
        return s[0]
    a = np.array([[(d_tot([i,l],[i+1,l])+d_tot([i,l],[i-1,l]))*(d_tot([i,l],[i,l+1])+d_tot([i,l],[i,l-1]))/4 for l in range(shp[1])] for i in range(shp[0])])
    a = a/np.sum(a)
    if sharpening:
        H = np.array([entropy(elt,a) for elt in s])
        H_0 = np.max(H)
    v = [np.ones(shp)]*k
    w = [np.ones(shp)]*k
    d = [np.zeros(shp)]*k
    
    # Creation of the (factorized) distance kernel
    M_freq = np.array([[np.exp(-d_freq(i,j)**2/gamma) for i in range(shp[1])] for j in range(shp[1])])
    M_temp = np.array([[np.exp(-d_time(i,j,lambd)**2/gamma) for i in range(shp[0])] for j in range(shp[0])])
    
    s = s.copy()
    
    for i in range(k):
        s[i] = s[i]/2 + a/2
    
    # Computing of the kernel convolution
    def ker(nu):
        rst = np.zeros(shp)
        for i in range(shp[0]):
            rst[i] = np.dot(M_freq,nu[i])
        rst = np.dot(M_temp,rst)
        return rst
    
    # Standard loop of the barycenter algorithm
    prev_mu = np.ones(shp)
    for _ in range(epoch_max):
        mu = np.ones(shp)
        with np.errstate(over='raise'):
            try:
                for i in range(k):
                    w[i] = s[i]/ker(a*v[i]) 
                    d[i] = v[i]*ker(a*w[i])   
                    mu = mu*(d[i]**alpha[i])
                if sharpening:
                    mu = entropic_sharpening(mu,H_0,a)
                for i in range(k):
                    v[i] = v[i]*mu/d[i]
            except:
                return 2*prev_mu-a
        prev_mu = mu.copy()
    return 2*mu-a

# Complete algo

def wass_bar(s,t,d,gamma=0.01,lambd=1,epoch_max=10,sharpening=True,compute_dur=False):
    """
        Inputs:
            s: list of float np.array of shape (n, m), list of velocities of n chords of m notes
            t: float np.array of shape (n), timings of each chord
            d: list of float np.array of shape (n, m), list of durations of n chords of m notes
            gamma: float, regularization float to make the optimization problem convex
            lambd: float, (distance on axis=0) = lamnd*(distance on axis=1)
            epoch_max: int, maximal number of loops to compute the barycenter
            sharpening: bool, True to use the entropic sharpening step of the Wasserstein barycenter
            compute_dur: bool, True to compute the Wasserstein barycenter of durations d
        Outputs:
            sbar: float np.array of shape (n, m), velocities of n chords of m notes, kernel Wasserstein barycenter of s
            tbar: float np.array of shape (n), timings of each chord equal to t
            dbar: float np.array of shape (n, m), durations of n chords of m notes, kernel Wasserstein barycenter of d id computed_dur=True, constant array of value 50 otherwise
    """
    alpha = np.ones(len(s))/len(s)
    sbar = ker_bar_exa(s,alpha,gamma=gamma,lambd=lambd,epoch_max=epoch_max,sharpening=sharpening)
    tbar = t
    if compute_dur:
        d_sum = np.array([np.sum(dur) for dur in d])
        for i in range(len(s)):
            d[i] = d[i]/d_sum[i]
        dbar = ker_bar_exa(d,alpha,gamma=gamma,lambd=lambd,epoch_max=epoch_max,sharpening=sharpening)*np.mean(d_sum)
    else:
        dbar = 50*np.ones(s[0].shape)
    return sbar,tbar,dbar