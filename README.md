# RICE_ts_class_training
This repository conains the code used to train and validate the ResNet neural networks (**Models 1-4**) used in the analysis workflow, **py4bleaching**. The related scripts for **py4bleaching** can be found in this [separate repository](https://doi.org/10.5281/zenodo.10616736). 

The data required to run the scripts found here is published [here](https://zenodo.org/records/10602864/) and can be downloaded using the ```data_download.py``` script within the ```src``` folder

This repository was made specifically for reference to a thesis chapter describing the optimisation of these models. 

## Prerequisites

This analysis assumes a standard installation of Python 3 (=> **3.7.10**). For specific package requirements, see the environment.yml file, or  create a new conda environment containing all packages by running ```conda create -f environment.yml```. 

## Data
Contains fluorescence trajectories used for various parts of the model training process. See ```README_data.md``` within ```data/``` folder for detailed descriptions of each file / subfolder within this folder. Brief directory provided below
| Folder      | Subfolder | Description   |
|-------------|----------------------|---------------|
|```0_raw_original_collated```|```1_AF488```| uprocessed trajectories from AF488 molecules only|
||```2_mixed```|uprocessed trajectories from AF488 & AF647 molecules|
|```1_specific-datasets```|```Model_1``` & ```Model_2```|original training dataset, and 'new' data it was tested on for figures (generating ROC/AUC curves)|
||```Model_3``` & ```Model_4```|```original``` training dataset for model 3 (extended to 1000 frames and filled with noise), and only ```new``` dataset for model 4, which was used to do transfer learning|

## Models
This contains the actual models generated by these data and scripts.

## Results
This contains the **OUTPUT** from each of the scripts in the src folder

## src
See ```src/README_src.md``` for detailed description/directory of scripts in each subfolder and their purpose. 
| Folder      |  Description   |
|-------------|---------------|
|```data_download.py```|This script must be run before anything else as it will download the data required to run the rest of the ```src``` scripts|
|```0_Training_model_scripts```| This is the scripts to be run in order to generate a classification model for classifying trajectories based on their shape, as well as the script that calls portions of these previous scripts as functions to run in one pipeline, and output a model. |
|```1_Optimising_generalisability```| These are scripts that generate datasets to optimise each of the new machine learning models described in **Figures 3.1-3.3** of this thesis.|


## Important note
This all pertains to the training and validating of a model for use in the ```py4bleaching``` pipeline. It is important to remember that the *'molecule name'* is saved so that once labels (classes) have been assigned, the labels can be mapped onto the raw data, as raw fluorescence is **REQUIRED** for all other steps in the analysis pipeline after classification.
