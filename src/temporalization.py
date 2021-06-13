import numpy as np
import scipy as sp
import math as m
import datetime, os
from os import walk

import MIDI_handling, sequence_alignment, wasserstein_barycenter, data_augmentation

def find_timing(pattern,s_tab,t_tab,transp_factor=0.2): # t_tab is made of absolute timings
    def dist(pat1,pat2):
        d_min = np.sum(pat1-pat2)
        for transp in range(-18,0): # -1.5 octave
            d = np.sum(pat1[:,-transp:]-pat2[:,:transp]) + abs(transp)*transp_factor
            d_min = min(d,d_min)
        for transp in range(1,19): # +1.5 octave
            d = np.sum(pat1[:,transp:]-pat2[:,:-transp]) + abs(transp)*transp_factor
            d_min = min(d,d_min)
        return d_min
    l = pattern.shape[0]
    d_min = dist(pattern,s_tab[0][:l])
    i_min = 0
    j_min = 0
    for i in range(len(s_tab)):
        for j in range(s_tab[i].shape[0]-l-1):
            d = dist(pattern,s_tab[i][j:j+l])
            if d<d_min:
                d_min = d
                i_min = i
                j_min = j
    return np.concatenate(([0],t_tab[i_min][j_min+1:j_min+l] - t_tab[i_min][j_min:j_min+l-1])) # return relative timings

def clean(seq,t,threshold=0.5): # return seq,t without blank chords, deleting each note with velocity<threshold
    seq = seq*(seq>=threshold)
    new_seq = []
    new_t = []
    for i in range(seq.shape[0]):
        if np.sum(seq[i]**2) > 0:
            new_seq += [seq[i]]
            new_t += [t[i]]
    new_seq = np.array(new_seq)
    new_t = np.array(new_t)
    return new_seq,new_t

def retempo(seq,t_loop,s_tab,t_tab,transp_factor=0.2,threshold=0.5): # we assume that t[0] and t[-1] != -1 (we don't oversample before 0 and after the end)
    """
        Inputs:
            seq: float np.array of shape (n, m), velocities of n chords of m notes
            t_loop: float np.array of shape (n), partial timings of each chord, -1 when not known, we assume that t_loop[0] and t_loop[-1] != -1
            s_tab: list of float np.array of shape (n_i, m), with n_i non necessarly constant, list of n_i chords of m notes (database)
            t_tab: list of float np.array of shape (n_i), with n_i non necessarly constant, list of n_i timings of chords (database)
            trans_factor: float, patterns are searched within a range of trans_factor*m in the database
            threshold: float, value under which velocities are rounded at 0
        Outputs:
            t: float np.array of shape (n), timings of each chord where the unknown timings of t_loop are found by similar pattern in the database (s_seq, t_seq)
    """
    seq,t = clean(seq,t_loop,threshold=threshold)
    pattern_begin = np.concatenate(((t!=-1)[:-1]*(t==-1)[1:],[False]))
    pattern_end = np.concatenate(([False],(t==-1)[:-1]*(t!=-1)[1:]))
    i_end = 0
    for i_beg in range(len(pattern_begin)):
        if pattern_begin[i_beg]:
            i_end = i_beg+1
            while i_end < len(pattern_end)-1 and not pattern_end[i_end]:
                i_end += 1
            if t[i_beg] == t[i_end]:
                for k in range(i_beg+1,i_end):
                    t[k] = t[i_beg]
            else:
                pattern = seq[i_beg:i_end+1]
                pattern_t = find_timing(pattern,s_tab,t_tab,transp_factor=transp_factor)
                pattern_t = np.cumsum(pattern_t)
                if pattern_t[0] == pattern_t[i_end-i_beg]:
                    for k in range(i_beg+1,i_end):
                        t[k] = t[i_beg]
                elif t[i_end] == -1:
                    for k in range(i_beg+1,i_end+1):
                        t[k] = pattern_t[k-i_beg]
                else:
                    pattern_t = pattern_t*(t[i_end]-t[i_beg])/(pattern_t[-1]-pattern_t[0]) + t[i_beg]
                    for k in range(i_beg+1,i_end):
                        t[k] = pattern_t[k-i_beg]
    return t