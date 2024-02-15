
# src directory
Directory of the scripts found within the ```src``` folder.

```0_Training_model_scripts/```

Overview of the steps to train a new model (taken for each of the four models)


**0**- *collect data* Collects data from multiple trajectories files and makes them into a similar format for later finding and concatinating

**1**- *cleanup trajectories* 
This makes them into a format which is appropriate for labelling

**2** - *prepare data for labelling*
This is where the  molecules need to be labelled manually in any way you prefer: this program plots the trajectories and allows you to classify them manually. Then, it will save the labelled trajectories for handling in python, and training the model

**3**- *train model*
Trains the actual model classifier which will be called on later

**4**- *Validate model*- this produces an ROC/AUC curve and plots comparisons for those that were not predicted correctly. 

*Important note*: the above scripts are the breakdown of developing the training pipeline. each of these are implemented within ```A_training```- this is a pipeline which runs through each of the above scripts as functions. However, they can each be run separately when troubleshooting or needing to use only a portion of the pipeline. 


# Main pipeline to train a new model:
If you don't need to run scripts separately due to troubleshooting/exploring the pipeline, the below scripts are the only ones necessary:

- ```0_collect_data.py```
```1_clean_trajectories.py``` - *always run these two scripts first*

- Then you can run...
```A_training.py``` - this has an if statement which stops and lets you label your trajectories. then goes on to train the model on the data once you have done so. It saves the best model after training. Importantly, this whole pipeline NORMALISES trajectories. This is because the non-normalising approach was least effective. 

- Finally, ```4_validate_model.py``` (generate ROC/AUC curves)


# Optimising Models (as described in Thesis Chapter 3)

```1_Optimising_generalisability/```

- This folder contains the scripts for preparing the data (i.e., manipulating the large, collated, manually labelled raw data so that only subsets required are used for training such as short only, long only etc.) and then re-training the new model / transferring weights. 

- In reality, ```a``` and ```b``` are built into ```A_training pipeline```, as well as validation, but this portion of scripts details how this was done, and if needed, can do this to data that doesn't need to pass through the entire pipeline. 
- Furthermore, the scripts in ```0_Training_model_scripts``` are more general, whilst these describe exactly what was done in this work, from a large body of trajectories.

```1_a_Model_2_normalising```

```1_b_Model_3_add_noise```

```1_c_Model_4_transfer_learning-```
-  This script should be run to make a new model with weights of a previous model. this means that this is not built into the ```'training'``` pipeline, as training is not necessary here. Validation is, though, and should be performed afterward. The results of this script are simply the original and newly saved model, and the data used was the 'new' data in the **Model 4** data folder of this repository.