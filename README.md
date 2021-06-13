# Musical Style Transfer project

## The project

The project consists in generating samples of "Les Sauvages, Rameau" from a database of different versions of the same piece with different ornements (in MIDI format), training a LSTM neural network with it.  

We project the chords from MIDI files (128 notes) into a smaller space using SVD fitted on MAESTRO 2017 dataset.  

This project is a prequel to the task of style transfer on MIDI files, using the MAESTRO dataset for classical style examples, and a database to be defined for the target style.  

The pipeline has 4 main steps:
* Atemporalization and alignment with dichotomic FastDTW,
* Approximate kernel Wasserstein barycenter,
* SVD projection and LSTM training/inference,
* Retemporalization with pattern finding.  

## Run

Data needed:
REF_DIR: directory containing the MAESTRO datasets (only the REF_DIR/2017 folder is used for SVD here)
TEST_DIR: test dataset directory, containing "Les Sauvages" midi files ("sauvages_full.mid" and "sauvages_sklt.mid", given in "./data"), and used to write the generated sequences
CHKP_DIR: checkpoint directory for logging

Run "python lstm_training.py" for a training+inference example.  
All the hyperparameters can be tuned in the main function of "lstm_training.py".  
