from mido import Message, MidiFile, MidiTrack
import numpy as np
import scipy as sp
import math as m

import datetime, os
from os import walk

def generate_seq_nump(file_name, len_max=-1, norm=True, time=False, duration=False, time_tolerance=0.010):
    """
        Inputs:
            db: str, path to the .midi file
            len_max: int, maximal length of the sequences, -1 for none
            norm: bool, True to normalize the velocities
            time: bool, True for returning timing array
            duration: bool, True for returning duration array
            time_tolerance: float, delta_time under which to consecutive notes are considered part of the same chord
        Outputs:
            s: float np.array of shape (n, m), velocities of n chords of m notes
            t: float np.array of shape (n), timings of each chord, returned if time=True
            d: float np.array of shape (n, m), durations of n chords of m notes, returned if duration=True
    """
    mid = MidiFile(file_name)

    s = []
    t = []
    d = []
    curr_time = 0
    prev_time = 0
    curr_notes = [0]*128

    for i, track in enumerate(mid.tracks):
        # print("Track {}: '{}' of length {}".format(i, track.name, len(track)))
        for mes in track:
            
            if mes.type == 'note_on' and (len(s) < len_max or len_max == -1):
                curr_time += mes.time
                if mes.velocity > 0:
                    if (curr_time-prev_time)*0.002253521 > time_tolerance:  # value linked to BPM
                        s += [[0]*128]
                        s[-1][mes.note] += float(mes.velocity)

                        d += [[0]*128]
                        d[-1][mes.note] -= curr_time

                        t += [curr_time]
                        prev_time = curr_time
                    else: #case of simultaneous notes
                        if len(s)>0:
                            s[-1][mes.note] += float(mes.velocity)
                            d[-1][mes.note] -= curr_time
                        else:
                            s += [[0]*128]
                            s[-1][mes.note] += float(mes.velocity)

                            d += [[0]*128]
                            d[-1][mes.note] -= curr_time

                            t += [curr_time]
                            prev_time = curr_time
                    curr_notes[mes.note] = len(s)-1
            if mes.type == 'note_off' and (len(s) < len_max or len_max == -1):
                curr_time += mes.time
                d[curr_notes[mes.note]][mes.note] += curr_time
            
            
    t += [curr_time] 
    s = np.array(s)
    t = np.array(t) - t[0]

    d = np.array(d)
    d = d*(d>=0) + 50*(d<0) # give a fixed duration to never-ended notes (it's the mean notes' duration un sauv_full)
    if norm:
        s /= np.sum(s) # to represent a distribution over the space time*freq
    else:
        s /= 128. # to map velocities in [0,1[
    rst = [s]
    if time:
        rst += [t.astype('float64')]
    if duration:
        rst += [d.astype(float)]
    if len(rst) == 1:
        return s
    return rst

def write_midi(v_seq,d_seq,t_seq,path,factor=5): # t_seq is a delta_time format, from real time t to delta time, just do t = t[1:]-t[:-1]
    """
        Inputs:
            s_seq: float np.array of shape (n, m), velocities of n chords of m notes
            t_seq: float np.array of shape (n), delta_timings of each chord (waiting time from previous chord to the current one)=
            d_seq: float np.array of shape (n, m), durations of n chords of m notes
            path: str, path to write the MIDI file
            factor: float, multip. factor on the timings
        Outputs:
            None
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    time = np.append([0],np.cumsum(t_seq))
    
    # Tempo correction
    time = time.astype('float64')
    time *= factor
    time = time.astype('int64')
    d_seq = d_seq.astype(float)
    d_seq *= factor
    d_seq = d_seq.astype(int)
    
    t = []
    mes = []
    v_max = np.amax(v_seq)
    
    for i in range(len(v_seq)):
        for j in range(len(v_seq[i])):
            if v_seq[i][j] > 0:
                mes += [['note_on',j,v_seq[i][j]]]
                t += [time[i]]
                
                mes += [['note_off',j,v_seq[i][j]]]
                t += [time[i]+d_seq[i][j]]
    zipped = list(zip(mes,t))
    zipped.sort(key=lambda itm:itm[1]+((itm[0][0]=='note_on')*1. - itm[0][2]*0.1)*0.1)
    
    t = []
    mes = []
    note_off = [False]*128
    note_on = [-1]*128 #stores timings of triggering
    for tup in zipped:
        if note_on[tup[0][1]] != tup[1] and tup[0][0] == 'note_on':
            note_on[tup[0][1]] = tup[1]
            mes += [tup[0]]
            t += [tup[1]]
        elif note_on[tup[0][1]] == tup[1] and tup[0][0] == 'note_on':
            note_on[tup[0][1]] = tup[1]
        else:
            mes += [tup[0]]
            t += [tup[1]]
    for i in range(len(t)-1,-1,-1):
        if note_off[mes[i][1]] and mes[i][0] == 'note_off':
            mes.pop(i)
            t.pop(i)
        elif note_off[mes[i][1]] and mes[i][0] == 'note_on':
            note_off[mes[i][1]] = True
    t = np.array(t)
    t = t[1:] - t[:-1]
    t = np.append([0],t)
    
    for i in range(len(mes)):
        track.append(Message(mes[i][0], note=mes[i][1], velocity=int(127*mes[i][2]/v_max), time=int(t[i])))

    mid.save(path)