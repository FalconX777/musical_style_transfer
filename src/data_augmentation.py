import numpy as np
import pandas as pd
import scipy as sp
import math as m
from scipy import signal
import datetime, os
import random
from os import walk

import MIDI_handling, sequence_alignment, wasserstein_barycenter

# Transposition

def transpo(s,min_tone=-18,max_tone=18):
    trans = random.randint(min_tone,max_tone) # +/- 1.5 octavas
    if trans>0:
        s_loop = [np.concatenate([np.zeros([s[i].shape[0],trans]),s[i][:,:-trans]],axis=1) for i in range(len(s))]
    elif trans<0:
        s_loop = [np.concatenate([s[i][:,-trans:],np.zeros([s[i].shape[0],-trans])],axis=1) for i in range(len(s))]
    else:
        s_loop = s
    return s_loop

# Chord split

def chord_split(s_loop,t_loop,p_split):
    s_sum = np.sum(np.array([np.sum(s_loop[i]>0,axis=1)>1 for i in range(len(s_loop))]), axis=1)
    split_index = np.random.random(s_loop[0].shape[0])
    proba_split = -(1-p_split)**s_sum + 1
    split_index = (split_index < p_split)
    index = np.cumsum(np.ones(s_loop[0].shape[0]))-1
    new_s = [np.zeros((0,128)) for _ in s_loop]
    new_t = []
    prev_i_split = 0
    for i_split in index[split_index]:
        i_split = int(i_split)
        for i in range(len(s_loop)):
            chord_1 = np.zeros((1,128))
            chord_2 = np.zeros((1,128))
            for j in range(128):
                vel = s_loop[i][i_split,j]
                if vel>0:
                    if np.random.normal(1)>0.5:
                        chord_1[0,j] = vel
                    else:
                        chord_2[0,j] = vel
            new_s[i] = np.concatenate([new_s[i],s_loop[i][prev_i_split:i_split],chord_1,chord_2])
        new_t = np.concatenate([new_t,t_loop[prev_i_split:i_split],[t_loop[i_split]],[t_loop[i_split]]])
        prev_i_split = i_split + 1
    for i in range(len(s_loop)):
        if prev_i_split < s_loop[i].shape[0]-1:
            new_s[i] = np.concatenate([new_s[i],s_loop[i][prev_i_split:]])
    if prev_i_split < t_loop.shape[0]-1:
        new_t = np.concatenate([new_t,t_loop[prev_i_split:]])
    return new_s,new_t

# Oversampling

def oversampling(s):
    return s

# Lambda finding

def find_density(s,i_max=8,j_max=10): # return the mean number of visible neighbors per active note, regarding the width and height of the filter
    s_bin = []
    for i in range(len(s)):
        s_bin += [s[i]>0]
    density = np.zeros([i_max,j_max])
    for seq in s_bin:
        for i in range(i_max):
            for j in range(j_max):
                filt = np.ones([2*i+1,2*j+1])
                density[i,j] += np.sum(seq*signal.convolve2d(seq, filt, mode='same'))/np.sum(seq)
    return density/len(s_bin)

exp_max = 7.45133*1e2
def kernel_ray(gamma,lambd):
    return np.sqrt(exp_max*gamma),np.sqrt(exp_max*gamma/lambd**2)

def kernel_param(i,j): # the kernel is of size (2*i+1,2*j+1)
    eps = 1e-3
    gamma = (i+1-eps)**2/exp_max
    lambd = np.sqrt(exp_max*gamma/(j+1-eps)**2)
    return gamma,lambd

def find_gamma(nb_neighbours,density,lambd=1):
    i_min = 0
    j_min = 0
    d_min = abs(density[0,0]-nb_neighbours)
    for i in range(density.shape[0]):
        j_lambd = 0
        lambd_min = abs(kernel_param(i,j_lambd)[1]-lambd)
        for j in range(density.shape[1]):
            loc_lambd = abs(kernel_param(i,j_lambd)[1]-lambd)
            if loc_lambd<lambd_min:
                lambd_min = loc_lambd
                j_lambd = j
        if abs(density[i,j_lambd]-nb_neighbours)<d_min:
            i_min = i
            j_min = j_lambd
            d_min = abs(density[i,j]-nb_neighbours)
    return kernel_param(i_min,j_min)[0]

def find_lambd(nb_neighbours,density,gamma=1):
    i_min = 0
    j_min = 0
    d_min = abs(density[0,0]-nb_neighbours)
    for i in range(density.shape[0]):
        j_gamma = 0
        gamma_min = abs(kernel_param(i,j_gamma)[0]-gamma)
        for j in range(density.shape[1]):
            loc_gamma = abs(kernel_param(i,j_gamma)[0]-gamma)
            if loc_gamma<gamma_min:
                gamma_min = loc_gamma
                j_gamma = j
        if abs(density[i,j_gamma]-nb_neighbours)<d_min:
            i_min = i
            j_min = j_gamma
            d_min = abs(density[i,j]-nb_neighbours)
    return kernel_param(i_min,j_min)[1]

def oversampling(s_loop,t_loop,oversampling_rule):
    return s_loop,t_loop

def find_approx_density(s,i_max=8,j_max=10,n_samples=100):
    s_bin = []
    for i in range(len(s)):
        s_bin += [s[i]>0]
    density = np.zeros([i_max,j_max])
    used_seq = 0
    for seq in s_bin:
        played_chord_index = np.cumsum(np.sum(seq,axis=1)>0)
        if played_chord_index[-1] > 0:
            used_seq += 1
            for _ in range(n_samples):
                chord_pos = random.randint(1,played_chord_index[-1])
                chord_index = chord_pos - 1
                while played_chord_index[chord_index] != chord_pos:
                    chord_index += 1
                note_pos = random.randint(1,np.sum(seq[chord_index]))
                note_index = -1
                while note_pos > 0:
                    note_index = (note_index + 1) % 128
                    if seq[chord_index,note_index] > 0:
                        note_pos -= 1
                for i in range(i_max):
                    for j in range(j_max):
                        chord_min = max(0,chord_index-i)
                        chord_max = min(seq.shape[0],chord_index+i+1)
                        note_min = max(0,note_index-j)
                        note_max = min(seq.shape[1],note_index+j+1)
                        density[i,j] += np.sum(seq[chord_min:chord_max,note_min:note_max])
                    
    if used_seq > 0:
        density /= n_samples*used_seq
    else:
        print('/!\ Empty sub-database')
    return density

# Velocity noise

def vel_noise(s_loop,mu_mean=0,mu_dev=1e-2,sigma_mean=1e-2,sigma_dev=1e-2):
    mu = np.random.normal(mu_mean,mu_dev,128)
    sigma = np.abs(np.random.normal(sigma_mean,sigma_dev,128))
    for i in range(len(s_loop)):
        s_loop[i] += np.random.normal(mu,sigma,s_loop[i].shape)
        s_loop[i] = np.abs(s_loop[i])
    return s_loop