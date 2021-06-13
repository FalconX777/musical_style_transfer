import numpy as np
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import scipy as sp
import math as m
import datetime, os
import random
from os import walk
import MIDI_handling, sequence_alignment, wasserstein_barycenter, data_augmentation, temporalization, lstm_training

def generate_seq(model, sbar_latent):
    num_generate = sbar_latent.shape[0]-n_forward
    sbar_tens = tf.convert_to_tensor(sbar_latent,tf.float32)
    seq_gen = np.zeros((0,latent_dim))

    predictions = tf.convert_to_tensor(svd.transform(np.zeros((1,128))),tf.float32)

    model.reset_states()
    for i in range(n_forward):
        input_eval = tf.expand_dims(tf.concat([predictions,sbar_tens[i:i+1]],1), 0)
        unused = model(input_eval)
        
    for i in range(n_forward,num_generate+n_forward):
        input_eval = tf.expand_dims(tf.concat([predictions,sbar_tens[i:i+1]],1), 0)
        
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # remove the batch dimension

        seq_gen = np.concatenate([seq_gen,predictions.numpy()])
    
    seq_gen = svd.inverse_transform(seq_gen)
    return seq_gen

def generate(model,out_path,min_var=0,max_var=31,var_set=np.ones(31)>0): #out_path contains the name
    """
        Inputs:
            model: LSTM trained model to generate a new sequence
            out_path: str, path to write the generated MIDI file
            min_var: int, minimum number of ornements per generated sample
            max_var: int, maximum number of ornements per generated sample
            var_set: bool np.array of shape (31), if var_set[i]==True, the i_th chunk of Les Sauvages can receive ornements, otherwise the standard chunk (without ornements) is always used
        Outputs:
            s: float np.array of shape (n, m), generated velocities of n chords of m notes
            t: float np.array of shape (n), generated timings of each chord
            d: float np.array of shape (n, m), constant durations of n chords of m notes
    """
    # Data generation and alignment
    print('Generating database',end='')
    s,t,d = gen_seq_db_sauv(n,sauvf,timef,durf,truncf,sauvs,times,durs,truncs,min_var=min_var,max_var=max_var,var_set=var_set,norm=False,time_tolerance=time_tolerance)
    ## Normalization of note velocity (not a real normalization, just the variance is set to 1e-2 (to have the average velocity of non-null note = 1), to keep the fact that 0 is the null-note for every chord)
    s_mean = [np.mean(s[i]) for i in range(len(s))]
    s_var = [np.mean(s[i]**2)-s_mean[i]**2 for i in range(len(s))]
    s = [s[i]/np.sqrt(s_var[i])/10 for i in range (len(s))]
    ## In the general case, the alignment has to be done after normalization, in the Sauvages case, generation and alignment are computed simultaneously for better efficiency (and there is no need of normalized input as there isn't any DTW)

    # Data augmentation, the training loop starts here, the modified s is s_loop
    t_loop = t.copy()
    s_loop = s.copy()
    random.shuffle(s_loop)
    ## Oversampling
    print('/oversampling',end='')
    s_loop,t_loop = oversampling(s_loop,t_loop,oversampling_rule)
    print('/lambda',end='')
    ## Find gamma for the barycenter (easier before adding noise)
    density = find_approx_density(s_loop,i_max=i_max,j_max=j_max,n_samples=n_samples)
    lambd = find_lambd(nb_neighbours,density,gamma=gamma)

    # Wasserstein Barycenter
    print(', computing barycenter',end='')
    ## Go back to initial distrib of velocities, and normalize the all distrib per track (sum==1)
    s_sum = [np.sum(s_loop[i]) for i in range(len(s_loop))]
    s_loop = [s_loop[i]/s_sum[i] for i in range(len(s_loop))]
    ## Compute the barycenter
    sbar,tbar,dbar = wass_bar(s_loop,t_loop,d,gamma=gamma,lambd=lambd,epoch_max=epoch_max,sharpening=sharpening,compute_dur=False)
    ## Barycenter post-processing
    sbar = sbar*(sbar>0)
    sbar = sbar/np.amax(sbar)
    sbar = sbar*(sbar>thr)
    ## Go back to normalized velocities per note
    sbar_mean = np.mean(sbar)
    sbar_var = np.mean(sbar**2)-sbar_mean**2
    sbar = sbar/np.sqrt(sbar_var)/10
    s_loop = [s_loop[i]*s_sum[i] for i in range(len(s_loop))]
    
    # Re Shaping
    sbar = np.concatenate([sbar,np.zeros((n_forward,128))])
    # Embedding
    s_latent = [svd.transform(seq) for seq in s_loop]
    sbar_latent = svd.transform(sbar)
    
    # Generation
    seq_gen = generate_seq(model, sbar_latent)
    
    # Retempo
    print(', retemporalization',end='')
    s_tab_clean = [s_loop[i][np.sum(s_loop[i],axis=1)!=0,:] for i in range(len(s_loop))]
    t_tab_clean = [t_loop[np.sum(s_loop[i],axis=1)!=0] for i in range(len(s_loop))]
    t = retempo(seq_gen,t_loop,s_tab_clean,t_tab_clean,transp_factor=0.2,threshold=0)

    ## Map the velocities between 0 and 127
    seq_gen -= np.amin(seq_gen)
    seq_gen *= 127/np.amax(seq_gen)
    seq_gen *= seq_gen>10

    # MIDI Writing
    print(', MIDI writing')
    dcstt = 50*np.ones(sbar.shape)
    t = t[1:]-t[:-1]
    write_midi(seq_gen,dcstt,t,out_path,factor=5)
    
    return seq_gen,t,dcstt