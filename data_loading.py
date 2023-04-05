"""This module contains functions to load, split, standardize and add stochasticity."""
import logging
import random
from collections import Counter


import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fastai.vision.all import *
from tsai.all import *
from tsai.imports import *
from tsai.utils import *
from tsai.data.core import TSDataLoaders, TSDatasets
from tsai.data.preprocessing import TSStandardize
from tsai.data.validation import get_splits


def set_random_seed(seed_value: int, dls=None, use_cuda: bool = True) -> None:
    """
    Set the random seed for various Python packages to ensure reproducibility.

    Parameters:
    -----------
    - seed_value (int): The seed value to be set.
    - dls (Optional[fastai.data.core.DataLoaders]): A `DataLoaders` object to set the random seed for.
                                                     Defaults to None.
    - use_cuda (bool): A boolean indicating whether CUDA (GPU) should be used. 
                       Defaults to True.

    Returns:
    --------
    None

    Notes:
    ------
    This function sets the random seed for the following packages:
        - numpy
        - PyTorch
        - Python's built-in random library
        - PyTorch GPU variables (if use_cuda=True)

    Additionally, if a `DataLoaders` object is provided, the random seed is set for it as well.
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if dls is not None:
        dls.rng.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random state set: {seed_value}, CUDA used: {use_cuda}")


def load_data(name, filepath, subset=-1):
    """
    Load input data and labels from files.

    Parameters:
    name (str): The name of the dataset.
    filepath (str): The file path where the input data and labels are stored.
    subset (int, optional): The number of samples to load from the dataset.

    Returns:
    X_raw (numpy array): The input data of shape (n_samples, n_features, n_timepoints).
    y (numpy array): The labels of shape (n_samples,).
    """

    # Validate input parameters
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(subset, int):
        raise TypeError("subset must be an integer")

    # Load input data
    try:
        X_raw = np.load(f"{filepath}/input_data/{name}_X.npy").astype(np.float32)
    except FileNotFoundError:
        logging.error(f"{name}_X.npy not found in {filepath}")
        return None, None

    # Load labels
    try:
        y_raw = np.load(f"{filepath}/input_data/{name}_YH.npy")
    except FileNotFoundError:
        logging.error(f"{name}_YH.npy not found in {filepath}")
        return None, None

    # y_test = np.expand_dims(y_raw[:, -1].astype(np.int64), -1)
    Y_raw = np.squeeze(y_raw)
    y = Y_raw[:, -1]

    # Subset data if required
    if subset > 0:
        X_raw = X_raw[:subset,:,:]
        y = y[:subset]

    print(f'Shape of X = {X_raw.shape}; Shape of y = {y.shape}')

    return X_raw, y

def onehot_encode(X):
    """
    One-hot encodes categorical features in the input data X.

    Args:
        X: A 3D numpy array of shape (num_samples, num_features, num_timepoints), where num_samples is the number of samples, 
            num_features is the number of features, and num_timepoints is the number of time steps.

    Returns:
        A 3D numpy array of shape (num_samples, num_oh_features, num_timepoints), where num_oh_features is the total number of one-hot
        encoded features after one-hot encoding all categorical features in X.
    """

    # Reshape the input array
    num_samples, num_features, num_timepoints = X.shape
    X_reshaped = X.transpose(0, 2, 1).reshape(-1, num_features)

    # One-hot encode the data
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X_reshaped).toarray()

    # Reshape the encoded array
    num_oh_features = X_encoded.shape[1]
    X_reshaped_encoded = X_encoded.reshape(num_samples, num_timepoints, num_oh_features).transpose(0, 2, 1)

    return X_reshaped_encoded

def standardize_data(X, splits):
    """
    Function to standardize the data using StandardScaler from scikit-learn. 
    Scales the train and validation/test sets separately based on the train set statistics.
    
    Parameters:
    - X: numpy array of shape (num_samples, num_features, num_timepoints) containing the input features
    - splits: tuple of numpy arrays containing the train, validation, and test indices
    
    Returns:
    - X_stnd: numpy array of shape (num_samples, num_features, num_timepoints) containing the standardized input features
    """
    scaler = StandardScaler()

    # Defining the dimensions
    num_samples, num_features, num_timepoints = X.shape

    # Fit the scaler encoder and scale the train data
    X_train = X[splits[0]]
    X_train = np.transpose(X_train, (0, 2, 1))
    X_train = np.reshape(X_train, (-1, num_features), order='A')
    X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train,  (len(splits[0]), num_timepoints, num_features))
    X_train = np.transpose(X_train, (0, 2, 1))

    # X_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1]))
    # X_train = X_train.reshape((len(splits[0]), X.shape[1], X.shape[2]))

    # Scale the valid data using the fitted scaler
    X_valid = X[splits[1]]
    X_valid = np.transpose(X_valid, (0, 2, 1))
    X_valid = np.reshape(X_valid, (-1, num_features), order='A')
    X_valid = scaler.transform(X_valid)
    X_valid = np.reshape(X_valid,  (len(splits[1]), num_timepoints, num_features))
    X_valid = np.transpose(X_valid, (0, 2, 1))

    # X_valid = scaler.transform(X_valid.reshape(-1, X.shape[-1]))
    # X_valid = X_valid.reshape((len(splits[1]), X.shape[1], X.shape[2]))

    # Combine the train and valid data
    X_stnd = np.zeros((num_samples, num_features, num_timepoints))
    X_stnd[splits[0]] = X_train
    X_stnd[splits[1]] = X_valid
    return X_stnd



def preprocess_data(X, splits):
    """
    Function to preprocess data by one-hot encoding and standardizing features.

    Parameters:
        X (numpy.ndarray): input data of shape (num_samples, num_features, num_timepoints)
        splits (tuple): A tuple containing the train and validation indices.

    Returns:
        - X_scaled (numpy.ndarray): preprocessed data of shape (num_samples, num_features', num_timepoints)
    """

    # Define which variables will be one-hot encoded
    oh_vars = [1, 2, 3, 4, 5, 6, 7, 8]
    num_samples, num_features, num_timepoints = X.shape
    stnd_vars = list(set(range(num_features)) - set(oh_vars))

    # One-hot encoding
    X_onehot = X[:, oh_vars, :]
    X_onehot_out=onehot_encode(X_onehot)

    # Standard scaling
    X_stnd = X[:, stnd_vars, :]
    X_stnd_out = standardize_data(X_stnd, splits)

    # Concatenate the one-hot encoded and standardized data
    X_scaled = np.concatenate([X_onehot_out, X_stnd_out], axis=1)
    return X_scaled


def split_data(X, y, randnum):
    """
    Splits the data into training and validation sets, and returns the split arrays.

    Parameters:
        X (numpy array): Input features
        y (numpy array): Target labels
        randnum (int): Random seed for reproducibility

    Returns:
        X_train (numpy array): Training set input features
        y_train (numpy array): Training set target labels
        X_valid (numpy array): Validation set input features
        y_valid (numpy array): Validation set target labels
    """

    # Split the data into training and validation sets with a 80/20 ratio
    splits = get_splits(
            y,
            valid_size=0.2,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum
            )
    X_trainvalid, X_test = X[splits[0]], X[splits[1]]
    y_trainvalid, y_test = y[splits[0]], y[splits[1]]

    print("Training & validation data shape:", X_trainvalid.shape, y_trainvalid.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
    print(f'Positive count in y = {Counter(y)}; Positive count in y_trainvalid = {Counter(y_trainvalid)}; Positive count in y_test = {Counter(y_test)}')

    return X_trainvalid, y_trainvalid, X_test, y_test, splits

def add_stochasticity(y, stoc_percent, randnum):
    """
    Adds stochasticity to a binary target vector Y by randomly flipping 
    stoc_percent * 100% of the positive class and negative class.

    Args:
        Y: A 1D numpy array of shape (num_samples,), where num_samples is the number of samples.
        stoc_percent: A float between 0 and 1 representing the percentage of positive 
              and negative samples to be randomly flipped.
        randnum: An integer used to set the random seed for reproducibility.

    Returns:
        A 1D numpy array of shape (num_samples,) with stochasticity added to the input Y.
    """

    # Set random seed
    set_random_seed(seed_value=randnum)

    # Get indices of 1s and 0s
    idx_ones = np.where(y == 1)[0]
    idx_zeros = np.where(y == 0)[0]

    # Select the number of positive values that need to be switched
    num_ones = idx_ones.shape[0]
    num_switch10 = int(np.ceil(stoc_percent*num_ones))

    # Randomly selects the correct number of 1s/0s that will be switched to 0s/1s
    idx_switch10 = np.random.choice(idx_ones, size=num_switch10, replace=False)
    idx_switch01 = np.random.choice(idx_zeros, size=num_switch10, replace=False)

    # Make a copy of the input array
    y_stoc = np.copy(y)

    # Add stochasticity
    y_stoc[idx_switch10] = 0
    y_stoc[idx_switch01] = 1
    return y_stoc

def stoc_data(y: np.ndarray, X: np.ndarray, stoc_percent: float, randnum: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], TSDataLoaders]:
    """
    Splits the input data into train, validation and test sets, adds stochasticity to the train and validation set,
    and creates and returns a `TSDataLoaders` object for the train and validation sets.

    Args:
        y (np.ndarray): 1D array with the target variable.
        X (np.ndarray): 3D array with the input data in the format (samples, variables, time steps).
        stoc_percent (float): The proportion of the positive values in the target variable that will be randomly changed.
        randnum (int): The seed for the random number generator.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], TSDataLoaders]: A tuple with the following elements:
        
        - X_new: 3D array with the input data for the train, validation and test sets after splitting and adding stochasticity.
        - X_new3d: The same as `X_new`, but reshaped to 3D format.
        - y_stoc: 1D array with the target variable for the train, validation and test sets after splitting and adding stochasticity.
        - yorg: 1D array with the original target variable before splitting and adding stochasticity.
        - splits_new: A tuple with 3 arrays representing the new splits for the train, validation and test sets.
        - dls: A `TSDataLoaders` object for the train and validation sets with the specified batch sizes and transforms.
    """

    # Split y into (train + validation) and test with 80:20 ratio
    splits = get_splits(y, valid_size=0.4, stratify=True, random_state=23, shuffle=True, test_size=0.0)

    # Copy data
    y_trainvalid = y[splits[0]].copy()
    X_trainvalid = X[splits[0]].copy()

    # Add stochasticity to (train + validation) together
    y_stoc_trainvalid = add_stochasticity(y_trainvalid, stoc_percent=stoc_percent, randnum=randnum)

    # Split (train + validation) into train and validation with 33.333333:66.666666
    sec_splits = get_splits(y_stoc_trainvalid, valid_size=0.33333333, stratify=True, random_state=randnum, shuffle=True, test_size=0.0)

    # Split into training and validation sets
    X_train = X_trainvalid[sec_splits[0]]
    X_valid = X_trainvalid[sec_splits[1]]
    X_test = X[splits[1]]
    y_train = y_trainvalid[sec_splits[0]]
    y_valid = y_trainvalid[sec_splits[1]]
    y_test = y[splits[1]]
    y_stoc_train = y_stoc_trainvalid[sec_splits[0]]
    y_stoc_valid = y_stoc_trainvalid[sec_splits[1]]

    # Concatenate arrays
    y_stoc = np.concatenate((y_stoc_train, y_stoc_valid, y[splits[1]]))
    yorg = np.concatenate((y_train, y_valid, y_test))
    X_new = np.concatenate((X_train, X_valid, X_test))
    X_new3d  =  to3d(X_new)

    # Define the new overall split (not just for train/validation)
    splits_new = get_splits(y_stoc, n_splits=1, valid_size=np.shape(y_valid)[0], test_size=np.shape(y[splits[1]])[0], shuffle=False)

    # Define transformations
    tfms = [None, [Categorize()]]

    # Define datasets and dataloaders
    dsets = TSDatasets(X_new3d, y_stoc, tfms=tfms, splits=splits_new, inplace=True)
    dls = TSDataLoaders.from_dsets(
                        dsets.train,
                        dsets.valid,
                        bs=[64, 128],
                        batch_tfms=[TSStandardize(by_var=True)],
                        num_workers=0,
                    )
    return X_new, X_new3d, y_stoc, yorg, splits_new, dls
