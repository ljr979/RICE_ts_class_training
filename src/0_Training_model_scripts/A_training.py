
import os
import re

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
from random import sample
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

#---------------------------------------

def prepare_data_for_labelling(input_files, output_folder, streamlit=False):

    if not os.path.exists (f'{output_folder}labelling_molecules/'):
        os.makedirs(f'{output_folder}labelling_molecules/')
    #
        
    #this is an if statement changes the input files depending on whether the molecules are labelled or not. if they've been labelled, the input files become a different path, and are called 'labelled data'
    
    if streamlit:
        input_files=[filename for filename in os.listdir(f'{output_folder}labelled_molecules/') if 'labelled_data' in filename]
    
    #this next chunk collects the trajectories and concatinate them, then giving them unique names and carrying the metadata from their original filename. 
        #this is done here regardless of labelling or not, as it needs to be done for both. then the saving of them changes depending on streamlit
    smooshed_trajectories=[]
    for filepath in input_files:
        trajectories=pd.read_csv(f'{output_folder}/labelled_molecules/{filepath}')
        trajectories.drop([col for col in trajectories.columns.tolist() if ' ' in col], axis=1, inplace = True)
        trajectories.rename(columns={'sample_name':'molecule_number'}, inplace=True)
        smooshed_trajectories.append(trajectories)
    smooshed_trajectories = pd.concat(smooshed_trajectories)
    #in case there are duplicate names we need to renumber (otherwise when classifying it it plots on top of one another)
    smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']]=smooshed_trajectories['molecule_number'].str.split('_', expand=True)
    smooshed_trajectories['number']=[str(x) for x in range(len(smooshed_trajectories))]
    smooshed_trajectories['molecule_number']=smooshed_trajectories[['treatment', 'colocalisation', 'protein', 'number']].agg('_'.join, axis=1)
    smooshed_trajectories.drop(['treatment', 'colocalisation', 'protein', 'number'], axis=1, inplace=True)
    
    #now if labelling has not occurred, this concatinated, labelled data needs to be SAVED as is (raw trajectories) and then the fluorescence needs to be NORMALISED to the maximum within each individual trajectory, and saved in a folder to be opened and lebelled in small chunks (100 trajectories per saved file). This subfolder will be called 'labelling molecules'
    if not streamlit:
        smooshed_trajectories.to_csv(f'{output_folder}labelling_molecules/smooshed_raw_data.csv')
        #normalisation section where we normalise to max value in each trajectory
        normalised_trajectories = smooshed_trajectories.copy().set_index('molecule_number')
        normalised_trajectories = (normalised_trajectories.T/normalised_trajectories.T.max()).T

        #split up into smaller chunks easier for streamlit stuff 
        n = 100  #chunk row size
        [normalised_trajectories[i:i+n].to_csv(f'{output_folder}labelling_molecules/data_for_training_{i}.csv') for i in range(0,normalised_trajectories.shape[0],n)]

    else:
        #this section accounts for x axis extending. Just extends and fills with NAN the extended length.
        #this is done only when labelling has already occurred, and is saved in the labelling molecules file
        #here we are just trying to make all of the data we are training on has 1000 time points so that if I have longer time series, the model we have trained can operate on that shape of data
        time_columns = [int(col) for col in smooshed_trajectories.columns.tolist() if col not in ['molecule_number', 'label']]
        if max(time_columns) != 1000:
            new_columns = [str(timepoint) for timepoint in range(max(time_columns)+1, 1000)]
            smooshed_trajectories[new_columns] = np.nan
        smooshed_trajectories.to_csv(f'{output_folder}labelling_molecules/smooshed_labels.csv')

def map_labels(input_path, output_folder, labels):
    # this function just maps any labels that a user might have manually assigned, to a binary classifier. This dictionary is defined in the main and may not even be necessary ,but allows for any manual inconsistencies. saves this as the file for training.
    labelled_data=pd.read_csv(input_path)
    labelled_data.drop([col for col in labelled_data.columns if 'Unnamed' in col], axis=1, inplace=True)
    labelled_data['label'] = labelled_data['label'].map(labels)
    #labelled_data.rename(columns={'sample_name':'molecule_number'}, inplace=True)
    labelled_data.to_csv(f'{output_folder}/labelling_molecules/labelled_data.csv')

#--------------------------------------------------------------
#this section here is the 'train_model script. this plots the data, splits up and preps train and test data for you, fits it with a classifier probability, then creates the actual classifier.
def plot_data_samples(dataframe, sample_numbers):
    ''' 
    Plot the time series data relating to the input list of sample numbers.

    sample_numbers: list of integers
        E.g. [1, 7, 22, 42]
    '''
    
    unique_labels = dataframe['label'].astype(int).unique()
    num_classes = len(unique_labels)
    if num_classes<=5:
        class_colors = dict(zip(unique_labels, ['palevioletred', 'crimson', 'purple', 'midnightblue', 'darkorange'][:num_classes]))
        palette = {sample: class_colors[class_number] for sample, class_number in dataframe.reset_index()[['index', 'label']].values}
    else:
        class_colors = sns.color_palette(n_colors=num_classes)

    for i in sample_numbers:
        logger.info(f'sample {i} is class {dataframe.loc[i, "label"].astype(int)}')

    for_plotting = pd.melt(dataframe.reset_index(), id_vars=['index', 'label'], var_name='time', value_vars=[col for col in dataframe.columns if col not in ['molecule_number', 'label']])
    for_plotting = for_plotting[for_plotting['index'].isin(sample_numbers)]
    fig, ax = plt.subplots()
    sns.lineplot(
        data=for_plotting,
        x='time',
        y='value',
        hue='index',
        palette=palette,
    )
    ax.set_ylabel('Data value')
    ax.set_xlabel('Timepoint')
    ticks = [x for x in range(for_plotting['time'].astype(int).max()) if x % 100 == 0 ]
    ax.set_xticks(ticks)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.savefig(f"{output_folder}samples.png")

def balance_data(time_data, labels):
    """[this function serves the purpose of balancing datasets that have been categorised for machine learning. It used SMOTE (see https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ for more info) in order to do this. SMOTE basically just over and undersamples severely unbalanced datasets so that the model can be built to pick trajectories on a balanced set, making it more reliable. OneHotEncodes categories so that the SMOTE function can do this, then tranfers this back into original categories using argmax :) iMPORTANT: my original dataset I made this fucntion for was 2700 trajectories. Thus, I have over and undersampled specifying the NUMBER of trajectories I want in each category originally this was 1000 of cateogries 1 and 0, and 500 of 2. THus is won't work for datasets that are super different to this. need to adjust this but for a similar ratio depending on your data (see link for more info on how many you should have in each category to be balanced). THis could be improved in the future to have this as something you enter into the function??? some way to make it a ratio rather than a number of trajectories? ]
    """
    #rename these so they're more similar to examples I found online just for my own confusion haha
    time_data = time_data.fillna(0)
    x_col = time_data
    y_col = labels
    

    #define one hot encoder 
    enc=preprocessing.OneHotEncoder(categories='auto')

    y_col = np.array(y_col)

    #run the onehotencoding on the y_cols array and reshape (think this is so it is a 2D array as it can't work on a 1d array)
    enc.fit(y_col.reshape(-1,1))

    #wouldn't run transform unless I turned this back into a dataframe because in the line below (taken directly from somethign you wrote earlier in this script) it can't use 'values' as .values needs a dataframe
    #y_col = pd.DataFrame(y_col)

    #transform data using the encoding (only works to a certain point when I have transferred back to dataframe, but doesn't run at all if I don't turn into dataframe)
    y_col = enc.transform(y_col.reshape(-1, 1)).toarray()


    #SMOTE method of balancing the data (top bit defines the pipeline and the parameters, bottom does the actual tranform)
    count = Counter(labels)
    over = SMOTE(sampling_strategy={2:400})
    under = RandomUnderSampler(sampling_strategy={0:800, 1:800})
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    x_col, y_col = pipeline.fit_resample(x_col, y_col)

    #trying to return original categorical label from Y_col back to labels
    y_true = np.argmax(y_col, axis=1)
    labels = pd.DataFrame(y_true)
    time_data = x_col
    #these lines convert the binary that was used for SMOTE back into original categories of the trajectories (0, 1 and 2). Now I just relabelled them something that is actually what we need to feed into the train test split which is then split into test and training, and fed into the model. 
    return time_data, labels

def train_new_model(time_data, labels, output_folder, itrs=1, classifier_name='resnet'):

    #this line balances the dataset by over and under sampling it so it's equal again (using SMOTE), then spits out equalised data for the splitting into test and train datasets
    logger.info(f'time data before balance {time_data.shape}')
    logger.info(f'labels before balance {labels.shape}')
    time_data, labels = balance_data(time_data, labels)
    logger.info(f'time data after balance {time_data.shape}')
    logger.info(f'labels after balance {labels.shape}')
    # Split data into train and test portions
    X_train, X_test, y_train, y_test = train_test_split(time_data, labels)

    # ----------------------train model----------------------

    for itr in range(itrs):
        output_directory = f'{output_folder}{classifier_name}_{itr+1}/'

        logger.info(f'Method: {classifier_name} using {itr+1} iterations.')

        if os.path.exists(f'{output_directory}df_metrics.csv'):
            logger.info(f'{classifier_name} using {itr+1} iteration already done')
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            fit_classifier(X_train, y_train, X_test, y_test, classifier_name, output_directory)

            logger.info('Training complete.')

            # evaluate best model on test data
            X_train, y_train, X_test, y_test = prepare_data_for_training(X_train, y_train, X_test, y_test,)
            model = keras.models.load_model(output_directory + 'best_model.hdf5')
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            model_metrics = calculate_metrics(y_true, y_pred, 0.0)
            logger.info(f'Iteration {itr+1}: df metrics')
            [logger.info(f'{measure}: {round(val, 2)}') for measure, val in model_metrics.T.reset_index().values]

def prepare_data_to_predict(time_data):
    if len(time_data.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        time_data = time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))
    return time_data

def predict_labels(time_data, model_path):
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(time_data)
    input_shape = x_predict.shape[1:]
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_predict)
    y_pred = np.argmax(y_pred, axis=1)
    # Add labels back to original dataframe
    time_data['label'] = y_pred
    return time_data

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
    data_subset = sample([col for col in data.columns.tolist()], 20)
    for molecule, df in comparison[comparison['diff'] == 1].groupby('molecule_number'):
        
        original_label = df['label'].tolist()[0]
        predict_label = df['predict_label'].tolist()[0]

        fig, ax = plt.subplots()
        sns.lineplot(data=data.groupby(['label', 'time']).mean().reset_index(), x='time', y='value', hue='label', palette=palette)
        sns.lineplot(x=np.arange(0, len(time_columns)), y=df[time_columns].values[0], color=palette[original_label], linestyle='--')
        plt.title(f'Molecule {molecule}: original label {original_label}, predicted {predict_label}')
        plt.show()

#--------------------


def pipeline(input_path,output_folder, labels):
    #this pulls together ALL of the functions above!
    #create folders for outputting
    output_folders = ['trained_model','validation']
    for folder in output_folders:
        if not os.path.exists(f'{output_folder}{folder}/'):
            os.makedirs(f'{output_folder}{folder}/')

    #-----------------------------
    #assign new labels to the classifiers, save this as 'labelled data'
    map_labels(input_path, f'{output_folder}', labels)
    #read in the labelled data again
    raw_data = pd.read_csv(f'{output_folder}labelled_molecules/labelled_data.csv')
    #drop this annoying column
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    #fill any NAns with 0
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data (no names, just fluorescence)
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]].reset_index(drop=True)

    # prepare label data as integers
    labels = raw_data['label'].copy().astype(int)

    # train model and save it
    train_new_model(time_data, labels, f'{output_folder}trained_model/', itrs=1, classifier_name='resnet')

    #read in raw data again
    raw_data = pd.read_csv(f'{output_folder}labelled_molecules/labelled_data.csv')
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    raw_data['label'] = raw_data['label'].fillna(0)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]]
    #now ask the model to predict the labels of the time series data (using the newly made model)
    time_data = predict_labels(time_data, f'{output_folder}trained_model/resnet_1/best_model.hdf5')
    time_data.groupby('label').count()
    #compare the labels column between the raw data (which has the manual labels) and the time data (which ahs the new labels assigned.)
    comparison = compare_labels(raw_data, time_data)
    palette = {0.0: 'firebrick', 1.0: 'darkorange', 2.0: 'rebeccapurple', '0': 'firebrick', '1': 'darkorange', '3': 'rebeccapurple'}
    #plot this comparison
    plot_comparison(comparison, palette=palette)

    

if __name__ == "__main__":
    #put the files here you'd like to use for labelling and subsequent training
    input_files = [
        'Results/training_model/clean_data/Exp1_cleaned_data.csv',
        'Results/training_model/clean_data/Exp2_cleaned_data.csv',
        ]
    #where would you like these to be saved
    output_folder = 'Results/training_model/compiled_trajectories/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #FIRST, run this function. This is BEFORE you've manually labelled the trajectories. it runs through 
    prepare_data_for_labelling(input_files=input_files, output_folder=output_folder, streamlit=False)

    #now do streamlit at this point and come back to run pipeline (now have a bunch of normalised trajectory files with same unique names because it's easier for streamlit, and just smoosh them back together for training model)
    #giving empty list for the input files because in the previous running of this function we define the input files as a bunch of files that have 'labelled_data' in them, which is what we get from strealit output
    prepare_data_for_labelling(input_files=[], output_folder=output_folder, streamlit=True)
    #now this is the new input path, because these labelled molecules have just been concatinated all together again.
    input_path= f'{output_folder}labelling_molecules/smooshed_labels.csv'

    #Example: dictionary accounting for differences in labelling, but mapping them back to be binary for classification by the model. 
    labels={'undefined':0, 'well defined':1, 'throw out':2}

    #make the model, and compare between manual and new labels
    pipeline(input_path, output_folder, labels)


