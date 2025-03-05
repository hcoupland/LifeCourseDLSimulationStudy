This respoitory is a companion to the paper "Exploring the Potential and Limitations of Deep Learning and Explainable AI for Longitudinal Life Course Analysis" and contains the code required to simulate data and run DL models and create the output plots shown in the paper.


## To generate the data:
Run the file Data_generation_DAG_gdm.R to generate 12 data sets with names Period1, Period2, etc.

## To run the models:
For each data set (Period1, Period2, ...) you run Model_run_postdoc/Main_run_file_gdm.py in the command line with the following arguments:
1. Device e.g. 1, 0 or 'cpu'
2. Data name, e.g. Period1, Timing3
3. Model name, e.g. XGBoost, LR, InceptionTime
4. Stochasticity, e.g. a number between 0 and 1 to give the number of positive cases switched
5. Random seed for training and hyperparameter optimisation
        (we conducted with 7, 8 and 9 and then averaged the results)

So in a command line this looks like:
    e.g. python Model_run_postdoc/Main_run_file_gdm.py 0 Period1 LSTMAttention 0.1 7

Then hyperparameter optimisation is conducted and the best model parameters are used for a final training.

## To collate the model run results:
Once you have run several models or stochasticity levels (for example), you can run the r script Results_collation/Results collation.R to gather all the different results into one table

## To plot the performance and identify the best model parameters:
Run the r script Plotting/Main_plotting.R, this outputs the plots for the paper, the table and prepares the data for explainability analysis.

## To conduct explainability anlaysis:
To obtain SHAP values and permutation feature importance (PFI) results, run the python script explr_from_outputs_gdm.py with the following arguments (that must match a completed run):
1. Device e.g. 1, 0 or 'cpu'
2. Data name, e.g. Period1, Timing3
3. Model name, e.g. XGBoost, LR, InceptionTime
4. Stochasticity, e.g. a number between 0 and 1 to give the number of positive cases switched
5. Random seed for training and hyperparameter optimisation
        (we conducted with 7, 8 and 9 and then averaged the results)
6. Whether to run feature importance/ calculate SHAP values, takes "True" or "False"

To plot the explainability results:
    Run the r script Plotting/XAI_plotting.R to obtain the plots given in the paper.

