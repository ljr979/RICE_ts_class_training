import os, re
import pandas as pd
import numpy as np
from loguru import logger
import glob
#cleans up individual trajectories which were put into a folder in this repository by the 0_collect_data script. as such, the output of that script is the input folder here.

input_folder = 'Results/collected_data/'
#This output will be where the cleaned up dataframe with all trajectories will save
output_folder = 'Results/training_model/clean_data/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)



def compile_trajectories(folder_list, input_folder, output_folder):
    #now loop through those folders  and pull out the CSV files (trajectories files) in each folder and put them in a dataframe with new columns to keep data about treatment 
    all_trajectory_data = []
    for folder in folder_list:
        #this line needs some sort of string formatting to fetch all the .csv files !!! only is taking the exact name of the folder and I can't figure it out. want to SPLIT on /*csv in all of my dummy data
        folder
        trajectories_files_list = [[f'{root}/{filename}' for filename in files if '.csv' in filename] for root, dirs, files in os.walk(f'{input_folder}{folder}')]
        #the below line is a confusing sentence that flattens the list from being multiple lists into one list
        trajectories_files_list = [item for sublist in trajectories_files_list for item in sublist]
        for filepath in trajectories_files_list:
            #changes the file path to replace weight back slashes that os.walk adds in, and makes hard to split on
            filepath = filepath.replace('\\', '/')
            file_details = re.split('/|\\\\', filepath)
            #this is the file name with all of the information
            exp_condition = file_details[-3].split('_')[-1]
            #this is the 'treatment' in the middle of the filename, between 'traj' and 'protein'
            treat_name = file_details[-2]
            #this is the protein, the first part of the filename
            protein_type = file_details [-1].split('_')[0]
            #this reads the entire path
            raw_trajectories = pd.read_csv(f"{filepath}")
            #drop the column that is read in (blank)
            raw_trajectories.drop([col for col in raw_trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)
            #transpose the dataframe to be a different orientation, and rename the index 'molecule_number' which is assigned by the image processing pipeline. We will rename these later to be more specific
            raw_trajectories = raw_trajectories.T.reset_index().rename(columns = {'index': 'molecule_number'})
            #assign new columns with the variables we just made
            raw_trajectories['treatment'] = exp_condition
            raw_trajectories['colocalisation'] = treat_name
            raw_trajectories['protein'] = protein_type
            raw_trajectories['file_path'] = filepath
            # store DataFrame in list
            all_trajectory_data.append(raw_trajectories)
        #stack these on top of each other to make a dtaframe with ALL of the trajctories for all of the files
    smooshed_trajectories = pd.concat(all_trajectory_data)
    #this is just so it matches something else :)
    smooshed_trajectories['colocalisation'] = smooshed_trajectories['colocalisation'].str.capitalize()

    #now need to output this file as the original data before renaming the molecules
    smooshed_trajectories.to_csv(f'{output_folder}/{folder}_initial_compiled_data.csv')
    return smooshed_trajectories

#find the folders withinthe input folder. this should be the 'proteins' which you defined in the previous script, which have another folder beneath them.
folder_list = [folder for folder in os.listdir(input_folder)]
#gather all trajectories and save initial compilation, before making new dataframe with new names
smooshed_trajectories=compile_trajectories(folder_list, input_folder, output_folder)

#now need to assign unique names to the molecules
smooshed_trajectories['metadata'] = smooshed_trajectories['treatment'] + '_' + smooshed_trajectories['colocalisation'] + '_' + smooshed_trajectories['protein']

#now we want to assign a UNIQUE and enumerated molecule number for every trajectory which carries the metadata through the entire analysis, so we can track where they came from.
smooshed_trajectories['molecule_number'] = [f'{metadata}_{x}' for x, metadata in enumerate(smooshed_trajectories['metadata'])]

timeseries_data = ['molecule_number'] + [col for col in smooshed_trajectories.columns.tolist() if type(col) == int]
timeseries_data = smooshed_trajectories[timeseries_data].copy()
#now save! for labelling
timeseries_data.to_csv(f'{output_folder}cleaned_data.csv')


