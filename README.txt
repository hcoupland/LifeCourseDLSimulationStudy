To generate the data:
    The file is Data_generation.R
    If you run it it should generate 13 data sets with names data1, data2, etc.

To run the ML model:
    For each data set (data1, data2, ...) you want to run one of each of the following files
    Deterministic Case:
        The file to run is Main_cmd0.py
        In the command line it takes the form: python Main_cmd0.py "DataSetName" "ModelName"
        Where ModelName is ResNet, InceptionTime, ResCNN, MLSTM-FCN, LSTM-FCN, XCM or TCN

    Stochastic Case:
        The file to run is Main_cmd0_stoc.py
        In the command line it takes the form: python Main_cmd0_stoc.py "DataSetName" "ModelName" proportion_of_noise
        Where ModelName is ResNet, InceptionTime, ResCNN, MLSTM-FCN, LSTM-FCN, XCM or TCN
        And proportion_of_noise is a number from 0.0 to 1.0 indciating the proportion of positive instance to swap with negatives
        I have just been running this with proportion_of_noise=0.1

I will add the Patch TST model in the morning and commit