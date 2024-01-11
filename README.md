# Train_validate_models
The repository containing scripts for training and validating Models 1-4 for py4bleaching. This repository was made specifically to correspond to the chapter in my thesis describing optimisation of these models. 

# DATA directory
0_raw_original_collated: this should contain all of the raw (no normalisation, no labels) trajectories from two different datasets. these files are called 'cleaned_data.csv'
1 = AF488-labelled chaperones, classified originally for model # 1
2= collated experiments with 488 and 647 labelled aggregation prone proteins and chaperones. this was used for models 3 and 4, and to evaluate models 1 & 2

1_specific_datasets
This folder contains subfolders with the specific subsets of trajectories used to train each model, and then to evaluate them and produce the ROC/AUC curves. The ones that were to train, have the labels that have been manually assigned to them in Streamlit, and they are in the form they need to be to train each model (i.e. normalised, x-axis extended, raw 488 fluorescence etc.). The new data, where applicable, is a dataset which is novel to the model, for evaluating outside of the train/test split. 

model 1:
original- non-normalised, AF488 labelled sHsp molecules. From one day of imaging. 600 frames long. (1600 trajectories)
new - non-normalised, AF488 and AF647 labelled sHsp and client (aggregate) molecule. from multiple imaging days. varying lengths. (2600 trajectories)

model 2:
original- normalised, AF488-and AF647 labelled sHsp and client (aggregate) molecules from multiple days of imaging. varying lengths (2500 trajectories)
new - normalised, AF488 labelled sHsp molecules. multiple lengths (1900 trajectories)

model 3:
original - normalised, AF488 and AF647 labelled sHsp and client (aggregate) moelcules from multiple imaging days. 600 frames extended to 1000 frames with random noise using the mean and stD of the prior 50 fluorescence values. (1950 trajectories). 
new - normalised, AF647 labelled client (aggregate) molecules from a single imaging day. all 900 frames long, extended to 1000 with random noise as in original. (300 trajectories)

model 4: 
original : model 2 weights, trained on short, normalised trajectories, 600 frames long. (see above)
new: normalised, AF647 labelled client (aggregate) molecules from a single imaging day. all 960 frames long(300 trajectories)


# Models
This contains the actual models generated by these data and scripts

# Results
This contains the OUTPUT from each of the scripts in the src folder

# Steps to train a new model (taken for each of the four models)

0_Training_model_scripts
0- collect data 
1- clean trajectories: this makes them into a format which is appropriate for labelling
2 - prepare data for labelling
- this is where the  molecules need to be labelled manually in any way you prefer: this program plots the trajectories and allows you to classify them manually. Then, it will save the labelled trajectories for handling in python, and training the model
3- train model
4- Validate model- this produces an ROC/AUC curve and plots comparisons for those that were not predicted correctly. 

important note: the above scripts are the breakdown of developing the training pipeline. each of these are implemented within A_training- this is a pipeline which runs through each of the above scripts as functions. However, they can each be run separately when troubleshooting or needing to use only a portion of the pipeline. 
MAIN TRAINING:
0_collect_data.py
1_clean_trajectories.py
then you can run...
A_training.py - this has an if statement which stops and lets you label your trajectories. then goes on to train the model on the data once you have done so. It saves the best model after training. Importantly, this whole pipeline NORMALISES trajectories. This is because the non-normalising approach was least effective. 
and finally, 4_validate_model.py (generate ROC/AUC curves)


1_Optimising_generalisability
in here, the scripts are for preparing the data (i.e., manipulating the large, collated, manually labelled raw data so that only subsets required are used for training such as short only, long only etc.) and then re-training the new model / transferring weights. In reality, a and b are is built into A_training pipeline, and validation but this details how this was done, and if needed, can do this to data that doesn't need dto pass through the entire pipeline. Furthermore, the scripts in 0_Training_model_scripts are more general, whilst these describe exactly what was done in this work, from a large body of trajectories.

1_a_Model_2_normalising
1_b_Model_3_add_noise
1_c_Model_4_transfer_learning


NOTE: MAY NEED TO ADD IN A SCRIPT WHICH BRIDGES THE GAP BETWEEN TRAINING THE MODEL ON DIFFERENT DATA (I.E., AFTER OPTIMISING GENERALISABILITY SCRIPTS) AND VALIDATING THESE NEW MODELS (I.E., TAKING THE RAW DATA, RUNNING IT BACK THROUGH THESE, KEEPING THE ORIGINAL LABELS AND BRINGING THE NEW ONES TOO). i think this is a combination between prep for comparing samples, and investigate_text_fix_training. 