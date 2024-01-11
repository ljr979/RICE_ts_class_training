import os, re
import pandas as pd
import numpy as np
from loguru import logger
import glob
import shutil
#This script takes the raw trajectory files and makes the name of the file consistent, and saved into the folder in the same place

#directory to where the 'raw' trajectories files have been output during image processing

input_folder = 'raw_data/'
#directory within the repository to save new copies of these files. 'Exp_1' is an example of a subfolder that would exist to show the specific experiment these raw files came from.
output_folder = 'Results/training_model/collected_data/Exp_1/'

#create output folder specified above
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#This is what the files will be named when they are moved from raw trajectories file, to this repository. Change these based on your experiment.
protein1_treatment1 = 'protein1_treatment1_traj'
protein2_treatment1 = 'protein2_treatment1_traj'
protein2_treatment2 = 'protein2_treatment2_traj'
protein1_treatment2 = 'protein1_treatment2_traj'


#change these for the treatments you're using
#this is a dictionary which defines the key as the location that the trajectories are now (specific folder underneath the input folder defined above), and the value is a tuple with (the location it is going to in this repository, the new trajectory name (the variable you defined above))
#this can be as many as you need, add as required
folders = {

'Exp_1_488_treat1/':('protein1/treatment1/', protein1_treatment1),
'Exp_1_488_treat1/':('protein2/treatment1/', protein2_treatment1),

'Exp_1_488_treat2/':('protein2/treatment2/', protein2_treatment2),
'Exp_1_488_treat2/':('protein1/treatment2/', protein1_treatment2),
    
    
    }



for old_folder, (new_folder, filetype) in folders.items():
    old_files = [filename for filename in os.listdir(f'{input_folder}{old_folder}') if '.csv' in filename]
    if not os.path.exists(f'{output_folder}{new_folder}'):
        os.makedirs(f'{output_folder}{new_folder}')
    for x, filename in enumerate(old_files): 
        shutil.copyfile(f'{input_folder}{old_folder}{filename}', f'{output_folder}{new_folder}{filetype}{x}.csv')


