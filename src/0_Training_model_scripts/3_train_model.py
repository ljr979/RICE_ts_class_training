"""A script that trains a new RESNET neural network for use on time-series trajectories (photobleaching)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tensorflow import keras
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import calculate_metrics
from sklearn.model_selection import GridSearchCV
from loguru import logger


def plot_data_samples(dataframe, sample_numbers):
    ''' 
    Plot the time series data relating to the input list of sample numbers.

    sample_numbers: list of integers
        E.g. [1, 7, 22, 42]
    '''
    unique_labels = dataframe['label'].astype(int).unique()
    num_classes = len(unique_labels)
    #defining colours to use based on the class
    if num_classes<=5:
        class_colors = dict(zip(unique_labels, ['palevioletred', 'crimson', 'purple', 'midnightblue', 'darkorange'][:num_classes]))
        palette = {sample: class_colors[class_number] for sample, class_number in dataframe.reset_index()[['index', 'label']].values}
    else:
        class_colors = sns.color_palette(n_colors=num_classes)

    for i in sample_numbers:
        logger.info(f'sample {i} is class {dataframe.loc[i, "label"].astype(int)}')

    #melt df for plotting. 
    for_plotting = pd.melt(dataframe.reset_index(), id_vars=['index', 'label'], var_name='time', value_vars=[col for col in dataframe.columns if col not in ['molecule_number', 'label']])
    #select the molecule name for the subset of each class
    for_plotting = for_plotting[for_plotting['index'].isin(sample_numbers)]
    #plot the intensity over time
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

def prepare_data(x_train, y_train, x_test, y_test,):
    """prepare the data to feed into the model architecture

    Args:
        x_train : output from train test split (training v testing datasets for validating model)
        y_train : output from train test split (training v testing datasets for validating model)
        x_test : output from train test split (training v testing datasets for validating model)
        y_test : output from train test split (training v testing datasets for validating model)

    Returns:
        train and test: the reshaped data for training
    """
    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test

def fit_classifier(x_train, y_train, x_test, y_test, classifier_name, output_directory):
    """creates the model architecture (classifier) and then fits your data to the architecture. saves the true values for each to use later.

    Args:
        x_train (array): x and y data for training and testing
        y_train (array): x and y data for training and testing
        x_test (array): x and y data for training and testing
        y_test (array): x and y data for training and testing
        classifier_name (str): classifier (type of model to use)
        output_directory (str): path to save model at
    """

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.values.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.values.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.values.reshape((x_test.shape[0], x_test.shape[1], 1))

    #define data shape to match to model
    input_shape = x_train.shape[1:]
    #create the model architecture
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    #fit data to model (train model)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    """imports the classifier you want to use for training

    Args:
        classifier_name (str): _description_
        input_shape (tuple): shape of the data output from fit_classifier function
        nb_classes (int): number of classes
        output_directory (str): output folder
        verbose (bool, optional):. Defaults to False.

    Returns:
        _type_: _description_
    """
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)

def train_new_model(time_data, labels, output_folder, itrs=1, classifier_name='resnet'):
    """Takes the timeseries data and trains a new model, and saves it

    Args:
        time_data (df): dataframe with the labelled time data
        labels (series): the possible labels from the dataset
        output_folder (str): where to save your new model
        itrs (int, optional): number of times to iterate over data for training. Defaults to 1.
        classifier_name (str, optional): _description_. Defaults to 'resnet'.
    """
    # Split data into train and test portions
    X_train, X_test, y_train, y_test = train_test_split(time_data.T, labels)

    # ----------------------train model----------------------
    #loop for as many iterations as requested, and train model
    for itr in range(itrs):
        output_directory = f'{output_folder}{classifier_name}_{itr}/'

        logger.info(f'Method: {classifier_name} using {itr} iterations.')

        if os.path.exists(f'{output_directory}df_metrics.csv'):
            logger.info(f'{classifier_name} using {itr} iteration already done')
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            fit_classifier(X_train, y_train, X_test, y_test, classifier_name, output_directory)

            logger.info('Training complete.')

            # evaluate best model on test data
            x_train, y_train, x_test, y_test = prepare_data(X_train, y_train, X_test, y_test,)
            model = keras.models.load_model(output_directory + 'best_model.hdf5')
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            model_metrics = calculate_metrics(y_true, y_pred, 0.0)
            logger.info(f'Iteration {itr}: df metrics')
            [logger.info(f'{measure}: {round(val, 2)}') for measure, val in model_metrics.T.reset_index().values]


if __name__ == '__main__':

    #input to the actual file with cleaned data ready for training (labelled)
    input_path ='tests/cleaned_data.csv'
    #where to save your new model
    output_folder ='results/test_dl4tsc/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #read in data
    raw_data = pd.read_csv(input_path)
    #drop columns not required
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)
    #The manually labelled data will only have labels on '1' and '2' data, so fill the unlabelled with '0', these should be the trajectories with undefined bleaching steps / exponential decay
    raw_data['label'] = raw_data['label'].fillna(0)


    # plot some sample data (take a sample of each classification to plot)
    sample_0s = raw_data.reset_index()[raw_data.reset_index()['label'] == 0]['index'].tolist()[:3]
    sample_1s = raw_data.reset_index()[raw_data.reset_index()['label'] == 1]['index'].tolist()[:3]
    sample_2s = raw_data.reset_index()[raw_data.reset_index()['label'] == 2]['index'].tolist()[:3]

    plot_data_samples(raw_data, sample_0s+sample_1s+sample_2s)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]].T.reset_index(drop=True)

    # prepare label data
    labels = raw_data['label'].copy().astype(int)

    # train model
    train_new_model(time_data, labels, output_folder, itrs=1, classifier_name='resnet')

