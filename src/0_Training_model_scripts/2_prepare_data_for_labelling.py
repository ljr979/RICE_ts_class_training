import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

input_files = [
    'Results/training_model/20201222_Exp_15_647data/clean_data/cleaned_data.csv',
    'Results/training_model/20210113_Exp_17_647data/clean_data/cleaned_data.csv'
    ]
output_folder = 'Results/training_model/20210525_647datacompiled/'

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

smooshed_trajectories=[]
for filepath in input_files:
    trajectories=pd.read_csv(f'{filepath}')
    trajectories.drop([col for col in trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)
    smooshed_trajectories.append(trajectories)
smooshed_trajectories = pd.concat(smooshed_trajectories)
#in case there are duplicate names we need to renumber (otherwise when classifying it it plots on top of one another)
smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']]=smooshed_trajectories['molecule_number'].str.split('_', expand=True)
smooshed_trajectories['number']=[str(x) for x in range(len(smooshed_trajectories))]
smooshed_trajectories['molecule_number']=smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']].agg('_'.join, axis=1)
smooshed_trajectories.drop(['treatment', 'colocalisation', 'protein', 'number'], axis=1, inplace=True)
#Run streamlit NOW on smooshed_trajectories df. 
smooshed_trajectories.to_csv(f'{output_folder}data_for_training.csv')
