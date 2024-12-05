To generate the data:
    The file is Data_generation_DAG_gdm.R
    If you run it it should generate 12 data sets with names Period1, Period2, etc.

To run the models:
    For each data set (Period1, Period2, ...) you run Model_run_postdoc/Main_run_file_gdm.py with the following arguments:
    1. Device e.g. 1, 0 or 'cpu'
    2. Data name, e.g. Period1, Timing3
    3. Model name, e.g. XGBoost, LR, InceptionTime
    4. Stochasticity, e.g. a number between 0 and 1 to give the number of positive cases switched
    5. Random seed for training and hyperparameter optimisation
        (we conducted with 7, 8 and 9 and then averaged the results)

    so in a command line this looks like:
        e.g. python Model_run_postdoc/Main_run_file_gdm.py 0 Period1 LSTMAttention 0.1 7

To collate the model run results:
    Run the r script Results_collation/Results collation.R to gather all the different results

To plot the performance and pick the best model parameters:
    Run the r script Plotting/Main_plotting.R, this outputs plots, the table and prepares the data for explainability analysis.

To conduct explainability anlaysis:
    Run the python script explr_from_outputs_gdm.py with the following arguments (that must match a completed run):
    1. Device e.g. 1, 0 or 'cpu'
    2. Data name, e.g. Period1, Timing3
    3. Model name, e.g. XGBoost, LR, InceptionTime
    4. Stochasticity, e.g. a number between 0 and 1 to give the number of positive cases switched
    5. Random seed for training and hyperparameter optimisation
        (we conducted with 7, 8 and 9 and then averaged the results)
    6. Whether to run feature importance/ calculate SHAP values, takes "True" or "False"

To plot the explainability results:
    Run the r script Plotting/XAI_plotting.R
