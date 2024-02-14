"""This script adds gaussian noise to the end of trajectories so that all are the same length. This is done here for normalised y-axis trajectories, of varying lengths such that they are all 1000 frames long.This dataframe can then be used to LABEL the molecules (manual classification) and then train using it.

Returns:
    dataframe with noise added to terminal end
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
def fill_nan(norm_traj):
    """fill empty cols with NAn

    Args:
        norm_traj (DF): dataframe with y normalised trajectories in it

    Returns:
        df: df with Nans at the empty timepoints (which have no fluorescence corresponding to that frame)
    """
    time_columns = [int(col) for col in norm_traj.columns.tolist() if col not in ['molecule_number']]
    if max(time_columns) != 1000:
        new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
        norm_traj[new_columns] = np.nan


    return norm_traj

def fill_noise_50(data, num_vals=50):
    """now to find the last 50 intensity values and average + SD of each molecule intensity. this results in the new DF being made with both SD and mean for every single molecule, which we can use to make a normal distribution to draw from when 

    Args:
        data (df): melted df with normalised trajectories filled to 1000 with NaN so all same length
        num_vals (int, optional): the number of values you want to use to fill the empty frames with, uses this many values to generate noise by finding mean and std. (of the preceding num of frames eg. the last 50 frames mean and std and noise around those values). Defaults to 50.
    """
    filled_data=[]
    for group, df in data.groupby(['molecule_number']):
        missing_values = df[df['intensity'].isnull()]
        complete_values = df[~df['intensity'].isnull()]
        last_fifty_av = complete_values.tail(num_vals).mean()['intensity']
        last_fifty_sd = complete_values.tail(num_vals).std()['intensity']
        missing_values['intensity'] = np.random.normal(last_fifty_av, last_fifty_sd, len(missing_values['intensity']))
        df = pd.concat([complete_values, missing_values])
        filled_data.append(df)

    filled_data=pd.concat(filled_data)
    return filled_data

if __name__ == "__main__":

    input_folder='Results/1_Optimising_generalisability/a_normalising/'
    output_folder='Results/1_Optimising_generalisability/b_add_noise/'
    #read in data
    norm_traj= pd.read_csv(f'{input_folder}cleaned_data.csv')
    #here we are just trying to make all of the data we are training on has 1000 time points so that if I have longer time series, the model we have trained can operate on that shape of data

    #extend trajectories
    data=fill_nan(norm_traj)
    #melt so that longform 
    data=pd.melt(data, id_vars=['molecule_number'], value_vars=[col for col in data.columns.tolist() if col not in ['molecule_number']], var_name='time', value_name='intensity')
    #fill with noise
    filled_data=fill_noise_50(data, num_vals=50)

    #now to unmelt the dataframe and save it to csv to be imported in my training script :) 
    fresh_training_data = filled_data.set_index(['molecule_number', 'time'])['intensity'].unstack().reset_index()
    #save so you can use as input to prepare_data_for_labelling or to train your model
    fresh_training_data.to_csv(f'{output_folder}cleaned_data.csv')

    #now plot a few to make sure they're added and look okay 
    data_subset = sample(list(filled_data['molecule_number'].unique()), 10)
    for molecule, df in filled_data.groupby(['molecule_number']):
        if molecule in data_subset: 
            df['time']=df['time'].astype(int)
            fig, ax = plt.subplots()
            sns.lineplot(data=df.groupby(['time']).mean().reset_index(), x='time', y='intensity')
            plt.title(f'Molecule {molecule}')
            plt.show()
