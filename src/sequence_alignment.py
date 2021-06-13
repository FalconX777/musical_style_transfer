import numpy as np
import scipy as sp
import math as m
import datetime, os
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from os import walk

import MIDI_handling

def gen_seq_db(db, len_max=-1, norm=True, time_tolerance=0.010):
    """
        Inputs:
            db: list of string, paths to .midi files
            len_max: int, maximal length of the sequences, -1 for none
            norm: bool, True to normalize the velocities
            time_tolerance: float, delta_time under which to consecutive notes are considered part of the same chord
        Outputs:
            s: list of float np.array of shape (n, m), list of velocities of n chords of m notes, aligned by dichotomic fast dtw
            t: float np.array of shape (n), timings of each chord equal to t, aligned by dichotomic fast dtw
            d: list of float np.array of shape (n, m), list of durations of n chords of m notes, aligned by dichotomic fast dtw
    """
    # generate the chord sequences
    s = []
    t = []
    d = []
    shuff_db = db.copy()
    random.shuffle(shuff_db)
    for path in shuff_db:
        s_i, t_i, d_i = generate_seq_nump(path, len_max=len_max, norm=False, time=True, duration=True, time_tolerance=time_tolerance)
        s += [s_i]
        t += [t_i.astype(float)]
        d += [d_i]
    
    # linear transfo to align the time of interp end
    mean_t = 0
    for i in range(len(t)):
        mean_t += t[i][-1]
    mean_t /= len(t)
    for i in range(len(t)):
        t[i] *= mean_t/t[i][-1]
    
    # align the sequences 2 by 2, building a binary tree, & map timestamps to each chord of the space
    curr_s = s.copy()
    children = [[i] for i in range(len(curr_s))]
    while len(curr_s) > 1:
        next_s = []
        new_children = []
        new_s = [[] for _ in s]
        new_t = [[] for _ in t]
        new_d = [[] for _ in d]
        for i in range(0,len(curr_s),2):
            if i+1<len(curr_s):
                # could be a good idea to use fastdtw on normalized chord sequences
                distance, path = fastdtw(curr_s[i], curr_s[i+1], dist=euclidean)
                last_tup = (-1,-1)
                last_tstmp = 0
                for tup in path:
                    # chord handling
                    if last_tup[0] != tup[0]:
                        for child in children[i]:
                            new_s[child] += [s[child][tup[0]]]
                            new_d[child] += [d[child][tup[0]]]
                    else:
                        for child in children[i]:
                            new_s[child] += [np.zeros(128)]
                            new_d[child] += [np.zeros(128)]

                    if last_tup[1] != tup[1]:
                        for child in children[i+1]:
                            new_s[child] += [s[child][tup[1]]]
                            new_d[child] += [d[child][tup[1]]]
                    else:
                        for child in children[i+1]:
                            new_s[child] += [np.zeros(128)]
                            new_d[child] += [np.zeros(128)]
                    
                    # time handling, simultaneous chunks only
                    if last_tup[0] != tup[0] and last_tup[1] != tup[1]:
                        # find median of timestamps
                        ttab = []
                        for child in children[i]:
                            ttab += [t[child][tup[0]]]
                        for child in children[i+1]:
                            ttab += [t[child][tup[1]]]
                        ttab.sort()
                        tstmp = ttab[len(ttab)//2]
                        
                        for child in children[i]:
                            new_t[child] += [max(last_tstmp,tstmp)]
                        for child in children[i+1]:
                            new_t[child] += [max(last_tstmp,tstmp)]
                        last_tstmp = tstmp
                    else:
                        for child in children[i]:
                            new_t[child] += [-1]
                        for child in children[i+1]:
                            new_t[child] += [-1]
                    last_tup = tup
                            
                # time handling, for non-simultaneous chunks
                j = 1
                j_0 = 0
                t_chunk_i = [0]
                t_chunk_ip = [0]
                while j<len(path):
                    if path[j][0] != path[j-1][0] and path[j][1] == path[j-1][1]:
                        # find median of timestamps
                        ttab = []
                        for child in children[i]:
                            ttab += [t[child][path[j][0]]-t[child][path[j-1][0]]]
                        ttab.sort()
                        tstmp = ttab[len(ttab)//2]
                        
                        t_chunk_i += [t_chunk_i[-1]+tstmp]
                        t_chunk_ip += [t_chunk_ip[-1]]
                        
                        j += 1
                    elif path[j][0] == path[j-1][0] and path[j][1] != path[j-1][1]:
                        # find median of timestamps
                        ttab = []
                        for child in children[i+1]:
                            ttab += [t[child][path[j][1]]-t[child][path[j-1][1]]]
                        ttab.sort()
                        tstmp = ttab[len(ttab)//2]
                        
                        t_chunk_ip += [t_chunk_ip[-1]+tstmp]
                        t_chunk_i += [t_chunk_i[-1]]
                        
                        j += 1
                    elif new_t[children[i][0]][j] > 0:
                        # dump t_chunk and recreate one
                        t_chunk_i += [new_t[children[i][0]][j]]
                        t_chunk_i = np.array(t_chunk_i)
                        t_chunk_i /= t_chunk_i[-1]
                        t_chunk_i *= new_t[children[i][0]][j] - new_t[children[i][0]][j_0]
                        
                        t_chunk_ip += [new_t[children[i+1][0]][j]]
                        t_chunk_ip = np.array(t_chunk_ip)
                        t_chunk_ip /= t_chunk_ip[-1]
                        t_chunk_ip *= new_t[children[i+1][0]][j] - new_t[children[i+1][0]][j_0]
                        t_chunk = [max(t_chunk_i[l],t_chunk_ip[l]) for l in range(len(t_chunk_i))]
                        for l in range(1,len(t_chunk)):
                            t_chunk[l] = max(t_chunk[l],t_chunk[l-1])
                        
                        if len(t_chunk) > 0:
                            t_chunk = np.array(t_chunk)
                            t_chunk += new_t[children[i][0]][j_0]
                            for k in range(len(t_chunk)):
                                for child in children[i]:
                                    new_t[child][j_0+k] = t_chunk[k]
                                for child in children[i+1]:
                                    new_t[child][j_0+k] = t_chunk[k]
                            
                        j_0 = j
                        j += 1
                        t_chunk_i = [0]
                        t_chunk_ip = [0]
                
                            
                nxt = np.zeros([len(new_s[children[i][0]]),128])
                for child in children[i]:
                    new_s[child] = np.array(new_s[child])
                    new_d[child] = np.array(new_d[child])
                    nxt += new_s[child]
                for child in children[i+1]:
                    new_s[child] = np.array(new_s[child])
                    new_d[child] = np.array(new_d[child])
                    nxt += new_s[child]
                nxt /= len(children[i]) + len(children[i+1])

                new_children += [children[i] + children[i+1]]
                next_s += [nxt]
            else:
                nxt = np.zeros([len(s[children[i][0]]),128])
                for child in children[i]:
                    new_s[child] = s[child]
                    new_t[child] = t[child]
                    new_d[child] = d[child]
                    nxt += new_s[child]
                nxt /= len(children[i])
                new_children += [children[i]]
                next_s += [nxt]
        s = new_s
        t = new_t
        d = new_d
        children = new_children
        curr_s = next_s
    
    
    if norm:
        for i in range(len(s)):
            s[i] /= np.sum(s[i])
    return s,(np.array(t[0])+0.5).astype(int),d


# Sauvages specific algorithm

def gen_seq_db_sauv(n,sauvf,timef,durf,sauvs,times,durs,min_var=0,max_var=10,var_set=np.ones(31)>0,norm=True,time_tolerance=0.010): # With time_tolerance = 0.010
    """
        Inputs:
            n: int, number of sequences to be generated
            sauvf: float np.array of shape (n1, m), velocities of n1 chords of m notes from sauv_full.mid (Les Sauvages with ornements)
            timef: float np.array of shape (n1), timings of n1 chords from sauv_full.mid (Les Sauvages with ornements)
            durf: float np.array of shape (n1, m), durations of n1 chords of m notes from sauv_full.mid (Les Sauvages with ornements) 
            sauvs: float np.array of shape (n1, m), velocities of n1 chords of m notes from sauv_sklt.mid (Les Sauvages without ornements)  
            times: float np.array of shape (n1), timings of n1 chords from sauv_sklt.mid (Les Sauvages without ornements)   
            durs: float np.array of shape (n1, m), durations of n1 chords of m notes from sauv_full.mid (Les Sauvages without ornements)   
            min_var: int, minimum number of ornements per generated sample
            max_var: int, maximum number of ornements per generated sample
            var_set: bool np.array of shape (31), if var_set[i]==True, the i_th chunk of Les Sauvages can receive ornements, otherwise the standard chunk (without ornements) is always used
        Outputs:
            s: list of float np.array of shape (n, m), list of velocities of n chords of m notes of random variations on Les Sauvages, exact alignment per chunk
            t: float np.array of shape (n), timings of each chord equal to t on Les Sauvages, exact alignment per chunk
            d: list of float np.array of shape (n, m), list of durations of n chords of m notes on Les Sauvages, exact alignment per chunk
    """
    truncf = np.array([  0,  24,  80, 121, 236, 239, 242, 262, 316, 327, 397, 408, 409,
                    414, 450, 457, 507, 524, 530, 532, 554, 640, 642, 654, 682, 683,
                    699, 722, 747, 759, 790, 830])
    truncs = np.array([  0,  16,  62,  89, 179, 180, 183, 198, 234, 243, 297, 305, 306,
                    310, 332, 339, 377, 388, 392, 393, 408, 463, 464, 473, 493, 494,
                    503, 518, 543, 552, 573, 608])
    s = []
    d = []
    
    var_full = np.zeros(var_set.shape)>0
    var_sklt = np.zeros(var_set.shape)>0
    var_set_tab = []
    
    # generation of boolean vectors choosing which variation to choose for each generated interp
    for _ in range(n):
        loc_var_set = var_set.copy()
        
        nb_var = random.randint(max(0,min_var),min(max_var,np.sum(var_set)))
        if nb_var > 0:
            rd = np.random.random(np.sum(var_set))
            thsd = np.sort(rd)[-nb_var]
            rd = rd>=thsd
        else:
            rd = np.zeros(var_set.shape)>0
        
        k = 0
        for i in range(loc_var_set.shape[0]):
            if loc_var_set[i]:
                loc_var_set[i] = rd[k]
                k += 1
            if loc_var_set[i]:
                var_full[i] = True
            else:
                var_sklt[i] = True
        var_set_tab += [loc_var_set]
    
    # alignment of each necessary chunk, depending if it will be used or not in the generation, and creation of the timestamp tab
    t = []
    aligned_sauvf = [[] for _ in range(var_set.shape[0])]
    aligned_sauvs = [[] for _ in range(var_set.shape[0])]
    aligned_durf = [[] for _ in range(var_set.shape[0])]
    aligned_durs = [[] for _ in range(var_set.shape[0])]
    for i in range(var_set.shape[0]):
        if var_full[i] and var_sklt[i]:
            kf = truncf[i]
            ks = truncs[i]
            while kf<truncf[i+1] or ks<truncs[i+1]:
                if timef[kf]==times[ks]:
                    t += [timef[kf]]
                    aligned_sauvf[i] += [sauvf[kf]]
                    aligned_sauvs[i] += [sauvs[ks]]
                    aligned_durf[i] += [durf[kf]]
                    aligned_durs[i] += [durs[ks]]
                    kf += 1
                    ks += 1
                elif timef[kf]<times[ks]:
                    t += [timef[kf]]
                    aligned_sauvf[i] += [sauvf[kf]]
                    aligned_sauvs[i] += [np.zeros(128)]
                    aligned_durf[i] += [durf[kf]]
                    aligned_durs[i] += [np.zeros(128)]
                    kf += 1
                else:
                    t += [times[ks]]
                    aligned_sauvf[i] += [np.zeros(128)]
                    aligned_sauvs[i] += [sauvs[ks]]
                    aligned_durf[i] += [np.zeros(128)]
                    aligned_durs[i] += [durs[ks]]
                    ks += 1
            aligned_sauvf[i] = np.array(aligned_sauvf[i])
            aligned_sauvs[i] = np.array(aligned_sauvs[i])
            aligned_durf[i] = np.array(aligned_durf[i])
            aligned_durs[i] = np.array(aligned_durs[i])
        elif var_full[i]:
            t += list(timef[truncf[i]:truncf[i+1]])
            aligned_sauvf[i] = sauvf[truncf[i]:truncf[i+1]]
            aligned_durf[i] = durf[truncf[i]:truncf[i+1]]
        elif var_sklt[i]:
            t += list(times[truncs[i]:truncs[i+1]])
            aligned_sauvs[i] = sauvs[truncs[i]:truncs[i+1]]
            aligned_durs[i] = durs[truncs[i]:truncs[i+1]]
    
    # generation of each interpretation using the aligned chunks
    for k in range(n):
        loc_var_set = var_set_tab[k]
        
        sauv_gen = np.zeros((0,128))
        dur_gen = np.zeros((0,128))
        for i in range(loc_var_set.shape[0]):
            if loc_var_set[i]:
                sauv_gen = np.concatenate((sauv_gen,aligned_sauvf[i]))
                dur_gen = np.concatenate((dur_gen,aligned_durf[i]))
            else:
                sauv_gen = np.concatenate((sauv_gen,aligned_sauvs[i]))
                dur_gen = np.concatenate((dur_gen,aligned_durs[i]))
        s += [sauv_gen]
        d += [dur_gen]
    
    if norm:
        for i in range(len(s)):
            s[i] /= np.sum(s[i])
    return s,np.array(t),d
