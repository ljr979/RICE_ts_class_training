import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
#there should now be 1 file called 'cleaned data' from 1_cleanup trajectories script, for each experiment, within its own subfolder of Results. 
#You need a list of these files to loop over if you are going to label a lot of trajetories from multiple experiments, as we will collate all of them together and give them new numbers so they don't accidentally duplicate names. 
input_folder='Results/training_model/clean_data/'
output_folder = 'Results/training_model/compiled_trajectories/'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)
#list, as a string, the name of the experiments subfolder (just below training_model, but above clean_data folders)
        

input_files=[filename for filename in os.listdir(input_folder) if 'cleaned_data.csv' in filename]

smooshed_trajectories=[]
for filename in input_files:
    trajectories=pd.read_csv(f'{input_folder}{filename}')

    trajectories.drop([col for col in trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)

    experiment_number=filename.split('_')[0]
    #keep the exp number as metadata temporarily
    trajectories['exp_num']=experiment_number
    smooshed_trajectories.append(trajectories)

smooshed_trajectories = pd.concat(smooshed_trajectories)
#in case there are duplicate names we need to renumber (otherwise when classifying it it plots on top of one another)
smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']]=smooshed_trajectories['molecule_number'].str.split('_', expand=True)
smooshed_trajectories['number']=[str(x) for x in range(len(smooshed_trajectories))]
smooshed_trajectories['molecule_number']=smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']].agg('_'.join, axis=1)
smooshed_trajectories.drop(['treatment', 'colocalisation', 'protein', 'number', 'exp_num'], axis=1, inplace=True)
#save this trajectories file! Then, open streamlit and it will plot all trajectories for you to classify. 
smooshed_trajectories.to_csv(f'{output_folder}data_for_training.csv')
