import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from loguru import logger
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dl4tsc.utils.constants import ARCHIVE_NAMES, CLASSIFIERS, ITERATIONS
from dl4tsc.utils.utils import calculate_metrics
from loguru import logger
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
#from src.python_photobleaching.analysis import clean_trajectories
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from random import sample
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where



def prepare_data_to_predict(raw_data, time_data, output_folder, model_path, x_norm=False):
    """prepares the data to be predicted by the model. this means that the data to be testing the model with needs to be in the same form as the model that was trained. 

    Args:
        raw_data (df): original manual labels, original data (raw)
        time_data (df): normalised, time data only
        output_folder (str): where to save
        model_path (str): where the model to be tested lives
        x_norm (bool, optional): how to alter the data. noise_50 fills any empty frames with gaussian noise using the last 50 values as mean and std. fillNAs fills them with NaN's (not so good). False leaves them as they are. They should already be normalised from clean_trajectories so this is not built in. Defaults to False.

    Returns:
        time_data: time data in the format required to be predicted on by the model
    """
    time_columns = [int(col) for col in time_data.columns.tolist()]
    #noise_50 keyword results in all trajectories bcoming the same length of time columns, then filling any missing values with random noise generated from the last 50 values of the trajectory. Then it spits out time_data with these filled values
    if x_norm == 'noise_50':
        if max(time_columns) != 1000:

            new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
            raw_data[new_columns] = np.nan
        #this chunk melts trajectories and turns it into 'data'. then data is grouped and the max value in thaat trajectory is found and turned into a new datafram called max_times. we then make a dictionary out of the molecule number (key) and the max time (value) and map this onto the original dataframe (data) to make a new column to say how long they are. 
        data=pd.melt(raw_data, id_vars=['molecule_number', 'label'], value_vars=[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']], var_name='time', value_name='intensity')
        #data=data.dropna(subset=['intensity'])
        #now to find the last 50 intensity values and average + SD of each molecule intensity. this results in the new DF being made with both SD and mean for every single molecule, which we can use to make a normal distribution to draw from when 
        filled_data=[]
        for group, df in data.groupby(['molecule_number', 'label']):
            missing_values = df[df['intensity'].isnull()]
            complete_values = df[~df['intensity'].isnull()]
            last_fifty_av = complete_values.tail(50).mean()['intensity']
            last_fifty_sd = complete_values.tail(50).std()['intensity']
            missing_values['intensity'] = np.random.normal(last_fifty_av, last_fifty_sd, len(missing_values['intensity']))
            df = pd.concat([complete_values, missing_values])
            filled_data.append(df)

        raw_data=pd.concat(filled_data)

        #now to unmelt the dataframe and save it to csv to be imported in my training script :) 
        raw_data = raw_data.set_index(['molecule_number', 'label', 'time'])['intensity'].unstack().reset_index()
        time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]]

        if len(time_data.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))

        #fillNAs takes the y normalised trajectories, and fills the empty spots with NAN'S which isn't going to be good but needed for validation. spits out time_data ready for prediction with NAns 
    elif x_norm == 'fillNAs': 
        if max(time_columns) != 1000:

            new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
            time_data[new_columns] = np.nan
            
        if len(time_data.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))

     

    #this one results in the data just staying as is! not adusting the x axis AT ALL. USe this version when you've created a new model with the short weights and new shape , or if you have the correct shape for an existing model already and don't need to change the x axis at all :) 

    if len(time_data.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))
    
    return time_data


def predict_labels(time_data, model_path, raw_data, output_folder, x_norm=False):
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(raw_data, time_data, output_folder, model_path, x_norm=x_norm)
    input_shape = x_predict.shape[1:]
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_predict)
    y_pred = np.argmax(y_pred, axis=1)
    # Add labels back to original dataframe
    time_data['label'] = y_pred
    return time_data

def ROC_AUC_probabilities(time_data, model_path, raw_data, output_folder, x_norm=False):
    """need to run this function as the validation at the end of the script, after the predict labels script to plot validation for each model.

    Args:
        time_data (df): _description_
        model_path (str): _description_
        raw_data (df): _description_
        output_folder (str): _description_
        x_norm (bool, optional): describes how the data is altered to match how the model which is being tested was trained. Defaults to False.

    Returns:
        _type_: _description_
    """
    x_predict = prepare_data_to_predict(raw_data, time_data, output_folder, model_path, x_norm=x_norm)
    
    input_shape = x_predict.shape[1:]
    model = keras.models.load_model(model_path)
    #returns probabilities to feed into the ROC/AUC graphing bit
    y_pred = model.predict(x_predict)

    return y_pred

def plot_ROC_AUC(raw_data, y_pred, output_folder):
    fpr = {}
    tpr = {}
    thresh ={}

    n_class = 3

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(raw_data['label'], y_pred[:,i], pos_label=i)
        
    # plotting    
    plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f'{output_folder}/Multiclass ROC',dpi=300);  

def plot_labels():

    data=pd.melt(time_data, id_vars='label', value_vars=[col for col in time_data.columns.tolist() if 'label' not in col], var_name='time')

    #plots the data in the colour for the label it was given
    data['time'] = data['time'].astype(int)
    sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label')
    sns.lineplot(data=data, x='time', y='value', hue='label')
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}

def compare_labels(raw_data, time_data):
    #compares the original label column with the predicted label column?
    raw_data['predict_label'] = time_data['label']
    raw_data['diff'] = [0 if val == 0 else 1 for val in (raw_data['label'] - raw_data['predict_label'])]
    return raw_data


def plot_comparison(comparison, palette=False):
    if not palette:
        palette='muted'

    time_columns=[col for col in comparison.columns.tolist() if col not in ['label', 'molecule_number','predict_label', 'diff']]

    data=pd.melt(comparison, id_vars=['label'], value_vars=time_columns, var_name='time')
    data['time']=data['time'].astype(int)
    for molecule, df in comparison[comparison['diff'] == 1].groupby('molecule_number'):
        
        original_label = df['label'].tolist()[0]
        predict_label = df['predict_label'].tolist()[0]

        fig, ax = plt.subplots()
        sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label', palette=palette)
        sns.lineplot(x=np.arange(0, len(time_columns)), y=df[time_columns].values[0], color=palette[original_label], linestyle='--')
        plt.title(f'Molecule {molecule}: original label {original_label}, predicted {predict_label}')
        plt.show()


if __name__ == "__main__":
    #path to model of interest
    model_path = f'Models/Model_3/model.hdf5'
    #path to trajectories you want to validate on (need to have manual labels too)
    input_path = f'data/1_specific_datasets/Model_3/original.csv'
    output_folder = f'Results/validating_model/Model_noise_added/original/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #use a descriptor here to describe the type of novel data you've tested your model on
    data_tested_on='long_trajectories'
    #read in raw data and time data
    raw_data = pd.read_csv(input_path)
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]]

    #CALCULATE the ROC/AUC for this model, based on comparison of the manual and predicted labels
    #use the keyword here that tells this function how you'd like the x-normalisation of the test data to be performed
    y_pred = ROC_AUC_probabilities(time_data, model_path, raw_data, output_folder, x_norm=False)
    #plots the validation
    plot_ROC_AUC(raw_data,y_pred,output_folder)

    #predict labels again on the time data
    time_data = predict_labels(time_data, model_path, raw_data, output_folder, x_norm=False)
    time_data.groupby('label').count()
    #compare the labels with the raw vs time data
    comparison = compare_labels(raw_data, time_data)
    #save this for reference
    comparison.to_csv(f'{output_folder}predicted_comparison.csv')


    #now calculate the accuracy and save this for reference
    incorrect=(len(comparison[comparison['diff']>0])/len(comparison))*100
    accuracy=[data_tested_on, incorrect]
    acc=pd.DataFrame(accuracy).T
    acc.columns=['data_for_ROC', 'incorrect_predictions (%)']
    acc.to_csv(f'{output_folder}accuracy.csv')


    #plot this comparison
    plot_comparison(comparison, palette='viridis')





