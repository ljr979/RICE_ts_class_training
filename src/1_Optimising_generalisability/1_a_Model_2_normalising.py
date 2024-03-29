"""This is a little script which shows how the raw fluorescence trajectories are normalised to the max fluorescence. This occurs within the other scripts, so is not necessary to do unless you require normalised trajectories of novel data etc. or are labelling some new molecules to test a model on.
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from random import sample
from matplotlib import pyplot
from numpy import where

#folder with the file output from 'cleanup trajectories', which should be RAW fluorescence values which need y-axis to be normalised per trajectory.
input_folder='data/0_raw_original_collated/2_mixed/'
output_folder='Results/1_Optimising_generalisability/a_normalising/'
#which experiment to target
exp_num = 'Exp1'
#read in data
raw_trajectories=pd.read_csv(f'{input_folder}{exp_num}_cleaned_data.csv')
raw_trajectories.drop([col for col in raw_trajectories.columns.tolist() if 'Unnamed: 0' in col], axis=1, inplace=True)
normalised_trajectories = raw_trajectories.copy().set_index('molecule_number')
#normalise to max fluoresscence of each molecule (i.e., normalise y axis)
normalised_trajectories = (normalised_trajectories.T/normalised_trajectories.T.max()).T
#save normalised data
normalised_trajectories.to_csv(f'{output_folder}cleaned_data.csv')
#this input can now be used in the 'prepare data for labelling' script prior to manual labelling prior to training. Or, this is done as part of the pipeline in A_training.py. 