import numpy as np
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import scipy as sp
import math as m
import datetime, os
import random
from os import walk
from sklearn.decomposition import TruncatedSVD

import MIDI_handling, sequence_alignment, wasserstein_barycenter, data_augmentation, temporalization, lstm_inference

# Dataset generator
# /!\ sub_seq_len = 1 for training, no working solution found for sub_seq_len > 0

def gen_batch(sauvf, timef, durf, sauvs, times, durs):
    def gen():
        while True:
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
            ## Transpose
            print(', transposition',end='')
            s_loop = transpo(s_loop,min_tone=min_tone,max_tone=max_tone)
            ## Chord split (np.mean(np.sum(sauvf>0,axis=1)) = 1.218)
            print('/splitting',end='')
            s_loop,t_loop = chord_split(s_loop,t_loop,p_split)
            ## Oversampling
            print('/oversampling',end='')
            s_loop,t_loop = oversampling(s_loop,t_loop,oversampling_rule)
            print('/lambda',end='')
            ## Find gamma for the barycenter (easier before adding noise)
            density = find_approx_density(s_loop,i_max=i_max,j_max=j_max,n_samples=n_samples)
            lambd = find_lambd(nb_neighbours,density,gamma=gamma)
            ## Velocity noise
            print('/noise',end='')
            s_loop = vel_noise(s_loop,mu_mean=mu_mean,mu_dev=mu_dev,sigma_mean=sigma_mean,sigma_dev=sigma_dev)

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
            
            # Reshape to fit the architecture
            s_loop = [np.concatenate([np.zeros((n_forward,128)),seq]) for seq in s_loop]
            sbar = np.concatenate([sbar,np.zeros((n_forward,128))])

            # Embedding
            s_latent = [svd.transform(seq) for seq in s_loop]
            sbar_latent = svd.transform(sbar)

            # Neural Network
            ## For LSTM training, inputs variables are s_latent,sbar_latent
            ## Hint: tf.convert_to_tensor(arg, dtype=tf.float32)
            print(', training LSTM')
            split = [int(i*(sbar.shape[0]-n_forward)/sub_seq_len) for i in range(sub_seq_len+1)]
            batch_sequence = [tf.convert_to_tensor([np.concatenate([seq[split[i]:split[i+1]+n_forward],sbar_latent[split[i]:split[i+1]+n_forward]],axis=1) for seq in s_latent],dtype=tf.float32) for i in range(sub_seq_len)]
            
            for batch in batch_sequence:
                input_batch = batch[:,:-n_forward,:]
                target_batch = batch[:,n_forward:,:latent_dim]
                yield input_batch, target_batch
    return gen

def build_model(batch_size,latent_dim):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(latent_dim, return_sequences=True, stateful=True,batch_input_shape=(batch_size,None,2*latent_dim)))
    return model

def train():
    """
        Training of a LSTM model over random variations of Les Sauvages, with data augmentation, using the list of hyperparameters listed in main.
        Outputs:
            model: the trained LSTM model
    """

    ##### SVD on MAESTRO #####
    direc = REF_DIR + '2017/'
    split = 0.8

    file_list = []
    for (d, ud, files) in walk(direc):
        file_list = files

    db = []
    for f in file_list:
        if f[-1] == 'i':
            db += [direc+f]
    random.Random(seed).shuffle(db)

    db_train = db[:int(split*len(db))]
    db_val = db[int(split*len(db)):]
    
    train_tab = [generate_seq_nump(filename, len_max=-1, norm=False, time=False, duration=False, time_tolerance=time_tolerance) for filename in db_train]
    val_tab = [generate_seq_nump(filename, len_max=-1, norm=False, time=False, duration=False, time_tolerance=time_tolerance) for filename in db_val]
    
    # Normalization of note velocity (not a real normalization, just the variance is set to 1e-2 (to have the average velocity of non-null note = 1), to keep the fact that 0 is the null-note for every chord)
    for i in range(len(train_tab)):
        train_mean = np.mean(train_tab[i])
        train_var = np.mean(train_tab[i]**2)-train_mean**2
        train_tab[i] = train_tab[i]/np.sqrt(train_var)/10
    
    for i in range(len(val_tab)):
        val_mean = np.mean(val_tab[i])
        val_var = np.mean(val_tab[i]**2)-val_mean**2
        val_tab[i] = val_tab[i]/np.sqrt(val_var)/10
    
    train_arr = np.concatenate(train_tab)
    val_arr = np.concatenate(val_tab)

    # Fitting the SVD
    svd = TruncatedSVD(n_components=latent_dim)
    svd.fit(train_arr)


    ##### MIDI Parsing #####
    sauvf, timef, durf = generate_seq_nump(sauv_full, -1, False, True, True, time_tolerance)
    sauvs, times, durs = generate_seq_nump(sauv_sklt, -1, False, True, True, time_tolerance)

    ds_batch = tf.data.Dataset.from_generator(
        gen_batch(sauvf, timef, durf,sauvs, times, durs), 
        output_types=(tf.float32,tf.float32), 
        output_shapes=((batch_size,None,2*latent_dim),(batch_size,None,latent_dim)))


    ##### Model building #####
    model = build_model(batch_size,latent_dim)
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])


    ##### Log Hyperparamaters #####
    dir_name = 'trans'+str(max_tone)+\
                '_split'*(p_split>0)+\
                ('_over'+str(over_mean))*(over_mean>0)+\
                '_noise'*(mu_dev>0)+\
                '_bar'+str(epoch_max)+\
                '_batch'+str(batch_size)+\
                '_epo'+str(epochs)+\
                '_stpepo'+str(steps_per_epoch)+\
                '_id'+str(random.randint(0,9999))+'/'
                
    os.mkdir(CHKP_DIR+dir_name)

    f = open(CHKP_DIR+dir_name+"hp.txt", "w")
    hp_str = ""
    hp_str += 'n = '+str(n)+'\n'
    hp_str += 'min_var = '+str(min_var)+'\n'
    hp_str += 'max_var = '+str(max_var)+'\n'
    hp_str += 'var_set = '+str(var_set)+'\n'
    hp_str += 'time_tolerance = '+str(time_tolerance)+'\n'
    hp_str += 'min_tone = '+str(min_tone)+'\n'
    hp_str += 'max_tone = '+str(max_tone)+'\n'
    hp_str += 'p_split = '+str(p_split)+'\n'
    hp_str += 'over_mean = '+str(over_mean)+'\n'
    hp_str += 'i_max = '+str(i_max)+'\n'
    hp_str += 'j_max = '+str(j_max)+'\n'
    hp_str += 'nb_neighbours = '+str(nb_neighbours)+'\n'
    hp_str += 'n_samples = '+str(n_samples)+'\n'
    hp_str += 'mu_mean = '+str(mu_mean)+'\n'
    hp_str += 'mu_dev = '+str(mu_dev)+'\n'
    hp_str += 'sigma_mean = '+str(sigma_mean)+'\n'
    hp_str += 'sigma_dev = '+str(sigma_dev)+'\n'
    hp_str += 'len_max = '+str(len_max)+'\n'
    hp_str += 'gamma = '+str(gamma)+'\n'
    hp_str += 'epoch_max = '+str(epoch_max)+'\n'
    hp_str += 'sharpening = '+str(sharpening)+'\n'
    hp_str += 'thr = '+str(thr)+'\n'
    hp_str += 'latent_dim = '+str(latent_dim)+'\n'
    hp_str += 'seed = '+str(seed)+'\n'
    hp_str += 'n_forward = '+str(n_forward)+'\n'
    hp_str += 'batch_size = '+str(n)+'\n'
    hp_str += 'epochs = '+str(epochs)+'\n'
    hp_str += 'steps_per_epoch = '+str(steps_per_epoch)+'\n'
    f.write(hp_str)
    f.close()

    ##### Train the model #####
    for epoch in range(epochs):
        print("Epoch ",epoch+1,"/",epochs, sep='')  
        if (epoch+1)%(epochs//nb_checkpts)==0:
            # Directory where the checkpoints will be saved
            checkpoint_dir = CHKP_DIR+dir_name
            # Name of the checkpoint files
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_"+str(epoch))
            checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)
            
            model.fit(ds_batch,epochs=1,steps_per_epoch=steps_per_epoch,callbacks=[checkpoint_callback])
        else:
            model.fit(ds_batch,epochs=1,steps_per_epoch=steps_per_epoch)
        model.reset_states()

    return model

if __name__ == '__main__':
    ##### Hyper parameters #####
    # directory containing the output .mid files
    REF_DIR = './datasets/Maestro/'             # directory containing the MAESTRO datasets (only the REF_DIR/2017 folder is used for SVD here)
    TEST_DIR = './datasets/Test_LSTM/midi/'     # test dataset directory, containing "Les Sauvages" midi files, and used to write the generated sequences
    CHKP_DIR = './LSTM_weights/'                # checkpoint directory for logging
    sauv_full = TEST_DIR + 'sauvages_full.mid'
    sauv_sklt = TEST_DIR + 'sauvages_sklt.mid'
    # Data generation
    n=1
    min_var=0
    max_var=5
    var_set=np.concatenate([np.ones(31),np.zeros(31-31)])>0
    time_tolerance=0.010
    # With time_tolerance = 0.010
    truncf = np.array([  0,  24,  80, 121, 236, 239, 242, 262, 316, 327, 397, 408, 409,
                    414, 450, 457, 507, 524, 530, 532, 554, 640, 642, 654, 682, 683,
                    699, 722, 747, 759, 790, 830])
    truncs = np.array([  0,  16,  62,  89, 179, 180, 183, 198, 234, 243, 297, 305, 306,
                    310, 332, 339, 377, 388, 392, 393, 408, 463, 464, 473, 493, 494,
                    503, 518, 543, 552, 573, 608])
    # Data augmentation
    ## Transposition
    min_tone = 0
    max_tone = 0
    ## Chord splitting
    p_split = 0.0 # probability of splitting a multi-note chord in 2 chords (random repartition between both chords)
    ## Oversampling
    over_mean = 0
    def oversampling_rule(size):
        return np.random.rayleigh(over_mean,size).astype(int)
    ## Find lambda
    i_max=10
    j_max=10
    nb_neighbours = 4
    n_samples = 100
    ## Vel_noise, for noise handling, for each frequency, the noise follows Normal(mu_freq,sigma_freq), with mu_freq that follows Normal(mu_mean,mu_dev) and sigma_freq following |Normal(sigma_mean,sigma_freq)|
    mu_mean = 0
    mu_dev = 1e-2
    sigma_mean = 1e-2
    sigma_dev = 1e-2
    # Wasserstein Barycenter
    ## Barycenter computing
    len_max = -1  
    gamma = 1  
    epoch_max = 20
    sharpening = False
    ## Post computing
    thr = 1e-1
    # Embedding
    embedding_rel_mse = 0.95
    latent_dim = 54
    seed = 9
    # LSTM 
    batch_size = n
    n_backward = 1
    n_forward = 1
    # Training
    epochs = 1600
    steps_per_epoch = 128
    sub_seq_len = steps_per_epoch
    nb_checkpts = epochs/5


    ##### Training #####
    model = train()


    ##### Generation #####
    generate(model,TEST_DIR+'generation.mid',min_var=min_var,max_var=max_var,var_set=var_set)